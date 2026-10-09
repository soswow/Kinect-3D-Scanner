"""Standalone server command; configuration is shared by Mac and Windows."""

import argparse
import os
import signal
import sys

from .exit_diagnostics import ExitDiagnostics


CHOICES = {
    "device": ("auto", "cpu", "cuda"),
    "tracking": ("auto", "legacy", "tensor"),
    "native": ("auto", "off", "on"),
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def port_number(value):
    number = positive_int(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def configure(argv=None):
    parser = argparse.ArgumentParser(description="Run the Kinect reconstruction server. Ctrl+C stops it.")
    parser.add_argument("--host", default=os.environ.get("KINECT_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=port_number, default=os.environ.get("KINECT_SERVER_PORT", "8000"))
    for name, choices in CHOICES.items():
        parser.add_argument(f"--{name}", choices=choices,
                            default=os.environ.get(f"KINECT_{name.upper()}", "auto"))
    parser.add_argument("--threads", type=positive_int, default=os.environ.get("OMP_NUM_THREADS", "4"))
    parser.add_argument("--block-count", type=positive_int, default=os.environ.get("KINECT_BLOCK_COUNT", "5000"))
    parser.add_argument("--max-frames", type=positive_int, default=os.environ.get("KINECT_MAX_FRAMES", "500"))
    args = parser.parse_args(argv)
    # argparse validates CLI choices; validate environment defaults as well.
    for name, choices in CHOICES.items():
        if getattr(args, name) not in choices:
            parser.error(f"--{name}: choose from {', '.join(choices)}")
    if not args.host.strip():
        parser.error("--host cannot be empty")
    for name, variable in (("host", "KINECT_SERVER_HOST"), ("port", "KINECT_SERVER_PORT"),
                           ("device", "KINECT_DEVICE"), ("tracking", "KINECT_TRACKING"),
                           ("native", "KINECT_NATIVE"), ("threads", "OMP_NUM_THREADS"),
                           ("block_count", "KINECT_BLOCK_COUNT"), ("max_frames", "KINECT_MAX_FRAMES")):
        os.environ[variable] = str(getattr(args, name))
    return args


def main(argv=None):
    args = configure(argv)
    from shared.diagnostics import configure_logging
    configure_logging(os.environ.get("KINECT_SERVER_LOG", "logs/server.log"))
    # Set native-thread/backend options before importing Open3D or the API.
    print(f"Starting Kinect server on {args.host}:{args.port}; device={args.device}, "
          f"tracking={args.tracking}, native={args.native}. Press Ctrl+C to stop.", flush=True)
    diagnostics = ExitDiagnostics(native_faults=True)
    diagnostics.record(f"Starting on {args.host}:{args.port}; device={args.device}, "
                       f"tracking={args.tracking}, native={args.native}")
    server = None
    try:
        import uvicorn

        class DiagnosticServer(uvicorn.Server):
            exit_signal = None

            def handle_exit(self, sig, frame):
                self.exit_signal = signal.Signals(sig).name
                diagnostics.record(f"Shutdown requested by {self.exit_signal}")
                super().handle_exit(sig, frame)

            async def shutdown(self, sockets=None):
                await super().shutdown(sockets=sockets)
                outcome = "failed" if getattr(self.lifespan, "shutdown_failed", False) else "completed"
                diagnostics.record(f"Shutdown {outcome}; reason={self.exit_signal or 'server requested exit'}")

        server = DiagnosticServer(uvicorn.Config("scanner_server.app:app", host=args.host, port=args.port))
        server.run()
        if not server.started and not server.exit_signal:
            diagnostics.record("Server exited before startup completed; exit_code=3")
            return 3
        diagnostics.record(f"Server loop returned; reason={server.exit_signal or 'server requested exit'}; exit_code=0")
    except KeyboardInterrupt:
        diagnostics.record("Server exited after keyboard interruption; exit_code=0")
        return 0
    except SystemExit as exc:
        diagnostics.record(f"Server raised SystemExit; exit_code={exc.code}")
        raise
    except Exception as exc:
        diagnostics.exception("Server failed with an unhandled exception; exit_code=1", exc)
        return 1
    finally:
        diagnostics.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
