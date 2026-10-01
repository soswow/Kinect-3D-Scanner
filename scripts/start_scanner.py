"""Run the scanner client and a local server; close both together."""

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
LOGS = ROOT / "logs"


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    LOGS.mkdir(exist_ok=True)
    port = int(os.environ.get("KINECT_SERVER_PORT", "8000"))
    # Do not interfere with a server or other application already on this port.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"Port {port} is already in use. Close the other scanner or set KINECT_SERVER_PORT.")

    env = os.environ.copy()
    env.update(KINECT_SERVER_HOST="127.0.0.1", KINECT_SERVER_PORT=str(port),
               KINECT_AUTOCONNECT="1", PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("KINECT_BLOCK_COUNT", "5000")
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    server = client = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    with (LOGS / "scanner-server.log").open("a") as server_log, \
            (LOGS / "scanner-client.log").open("a") as client_log:
        try:
            print(f"Starting local scanner server on 127.0.0.1:{port}...", flush=True)
            server = subprocess.Popen([sys.executable, "-m", "scanner_server"],
                                      cwd=PROJECT, env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 90
            with httpx.Client(trust_env=False, timeout=1) as http:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited. See {LOGS / 'scanner-server.log'}")
                    try:
                        response = http.get(f"http://127.0.0.1:{port}/api/health")
                        response.raise_for_status()
                        if response.json().get("status") == "ok":
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Server startup timed out. See {LOGS / 'scanner-server.log'}")
                    time.sleep(0.25)

            print("Server ready. Opening Kinect 3D Scanner. Close its window to stop both processes.", flush=True)
            print(f"Logs: {LOGS / 'scanner-client.log'} and {LOGS / 'scanner-server.log'}", flush=True)
            client = subprocess.Popen([sys.executable, "-m", "kinect_scanner"],
                                      cwd=PROJECT, env=env, stdout=client_log,
                                      stderr=subprocess.STDOUT)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Server stopped unexpectedly. See {LOGS / 'scanner-server.log'}")
                time.sleep(0.25)
            if client.returncode:
                raise RuntimeError(f"Client exited with code {client.returncode}. See {LOGS / 'scanner-client.log'}")
        except KeyboardInterrupt:
            print("Stopping scanner...", flush=True)
        finally:
            stop(client)
            stop(server)


if __name__ == "__main__":
    main()
