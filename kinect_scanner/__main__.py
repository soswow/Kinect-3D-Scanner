"""Entry point: python -m kinect_scanner"""

import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import subprocess
import sys


def main():
    # Frozen children reuse the app executable. Divert them before Qt/Open3D.
    multiprocessing.freeze_support()
    app = None
    from .runtime import log_path

    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3)]
        if sys.stderr is not None:
            handlers.append(logging.StreamHandler())
        logging.basicConfig(level=logging.INFO, handlers=handlers,
                            format="%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s")
        if len(sys.argv) > 1 and sys.argv[1] == "--viewer":
            from .viewer import _main

            return _main(sys.argv[2:])
        if "--check" in sys.argv[1:]:
            from .startup_check import run_check

            return run_check()

        from PyQt6.QtWidgets import QApplication, QMessageBox
        from .gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("Kinect 3D Scanner")
        app.setOrganizationName("Kinect3DScanner")
        app.setStyle("Fusion")

        def report_exception(kind, value, traceback):
            logging.getLogger(__name__).error("Unhandled client error", exc_info=(kind, value, traceback))
            QMessageBox.critical(None, "Kinect 3D Scanner", f"{value}\n\nDetails: {path}")

        sys.excepthook = report_exception
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as exc:
        logging.getLogger(__name__).exception("Client startup failed")
        message = f"The scanner could not start: {exc}\n\nDetails: {path}"
        if app is not None:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(None, "Kinect 3D Scanner", message)
        elif sys.platform == "darwin" and "--check" not in sys.argv:
            # Qt itself may have failed to import. Pass text as an argument, not code.
            try:
                subprocess.run(["/usr/bin/osascript", "-e",
                                'on run argv\n display alert "Kinect 3D Scanner" message (item 1 of argv) as critical\nend run',
                                message], check=False)
            except OSError:
                pass
        if sys.stderr is not None:
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
