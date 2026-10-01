"""Entry point: python -m scanner_server"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "14")

import uvicorn  # noqa: E402

uvicorn.run(
    "scanner_server.app:app",
    host=os.environ.get("KINECT_SERVER_HOST", "0.0.0.0"),
    port=int(os.environ.get("KINECT_SERVER_PORT", "8000")),
)
