"""Bounded, timestamped operational logs and out-of-band resource sampling."""

import json
import logging
import shutil
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(path):
    path = Path(path).resolve()
    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(path.resolve()) for h in root.handlers):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)sZ pid=%(process)d thread=%(threadName)s %(levelname)s %(name)s: %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stderr is not None and not any(type(h) is logging.StreamHandler for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger(__name__).info("Process started executable=%s log=%s", sys.executable, path)


class ResourceMonitor:
    """Keep sampling during GUI stalls and blocking build/download operations."""

    def __init__(self, disk_path, interval=15):
        self.disk_path = Path(disk_path)
        self.interval = interval
        self._stop = threading.Event()
        self._disk_low = False
        self._thread = threading.Thread(target=self._run, name="Resource monitor", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)

    def sample(self):
        result = {"disk_free_mib": round(shutil.disk_usage(self.disk_path).free / 1024**2)}
        try:
            import psutil
            process = psutil.Process()
            result["rss_mib"] = round(process.memory_info().rss / 1024**2, 1)
            result["system_available_mib"] = round(psutil.virtual_memory().available / 1024**2)
            result["swap_used_mib"] = round(psutil.swap_memory().used / 1024**2)
            children = []
            for child in process.children(recursive=True):
                try:
                    children.append({"pid": child.pid, "name": child.name(),
                                     "rss_mib": round(child.memory_info().rss / 1024**2, 1)})
                except psutil.Error:
                    pass
            result["children"] = children
        except ImportError:
            result["memory_unavailable"] = "Install psutil for memory sampling"
        logging.getLogger(__name__).info("Resources %s", json.dumps(result, sort_keys=True))
        disk_low = result["disk_free_mib"] < 1024
        if disk_low and not self._disk_low:
            logging.getLogger(__name__).warning("Low disk space: %s MiB free", result["disk_free_mib"],
                                                extra={"ui_state_key": "disk-space"})
        elif self._disk_low and not disk_low:
            logging.getLogger(__name__).info("Disk space recovered: %s MiB free", result["disk_free_mib"],
                                             extra={"ui_event": True, "ui_state_key": "disk-space"})
        self._disk_low = disk_low
        return result

    def _run(self):
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                logging.getLogger(__name__).exception("Resource sampling failed")
            self._stop.wait(self.interval)
