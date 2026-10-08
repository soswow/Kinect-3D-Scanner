"""Executable entry point; child-process dispatch happens before GUI imports."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from kinect_scanner.__main__ import main

    raise SystemExit(main())
