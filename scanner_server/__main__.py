"""Standalone server command; configuration is shared by Mac and Windows."""

import argparse
import os
import sys

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
    try:
        import uvicorn

        uvicorn.run("scanner_server.app:app", host=args.host, port=args.port)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Server startup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
