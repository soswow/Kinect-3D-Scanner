<p align="center">
  <h1 align="center">Kinect 3D Scanner</h1>
  <p align="center">
    A client/server application for 3D scanning with the Xbox 360 Kinect (v1).<br/>
    Capture frames on a laptop, process on a server, export to Blender-compatible PLY/OBJ.
  </p>
</p>

<p align="center">
  <a href="#run-client-and-server-on-one-machine"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue" alt="Platform"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.10+-yellow?logo=python&logoColor=white" alt="Python"></a>
  <a href="#usage"><img src="https://img.shields.io/badge/UI-PyQt6-green?logo=qt&logoColor=white" alt="PyQt6"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/server-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="docs/KINECT_V1_LINUX_PYTHON_REFERENCE.md"><img src="https://img.shields.io/badge/docs-reference-orange?logo=readthedocs&logoColor=white" alt="Docs"></a>
</p>

---

## Run Client and Server on One Machine

The client and server can run on the same computer over loopback HTTP and
WebSocket connections. Install the dependencies from `requirements.txt` into a
Python environment with working `freenect` bindings, then run from the repository:

```bash
python scripts/start_scanner.py
```

The launcher starts a server on `127.0.0.1:8000`, waits for its health check,
opens the GUI, and connects automatically. Closing the GUI stops the server.
Only one application should use the Kinect at a time.

On macOS, **Start Scanner.command** can be opened from Finder. It uses the
repository's `.venv/`, an activated virtual environment, or `python3` from PATH.
For an environment located elsewhere, run the launcher with that environment's
Python interpreter directly. The driver and compiled Python bindings must both
be installed; the Python dependencies alone do not provide `freenect`.

The software has been checked on Apple Silicon macOS with Python 3.13 and
Open3D 0.20. A tested dependency snapshot is in `requirements-macos-lock.txt`:

```bash
python -m pip install -r requirements-macos-lock.txt
```

This snapshot excludes `freenect`, which must be compiled separately. Synthetic
checks cover reconstruction and the client workflow; live capture and mesh
building have also been exercised on Kinect v1 hardware. Reconstruction quality
depends on overlap, camera calibration, and the subject's geometry.

The launcher uses four OpenMP threads and an initial 5,000-block TSDF allocation
(roughly 400 MB of voxel attributes, plus overhead). Larger scans may need more
blocks. These environment variables can override its defaults:

| Variable | Launcher default | Purpose |
|----------|------------------|---------|
| `KINECT_SERVER_PORT` | `8000` | Local server and client port |
| `KINECT_BLOCK_COUNT` | `5000` | Initial TSDF block budget |
| `KINECT_MAX_FRAMES` | `500` | Stored-frame limit per session |
| `OMP_NUM_THREADS` | `4` | CPU processing threads |

```bash
KINECT_SERVER_PORT=8001 KINECT_BLOCK_COUNT=10000 python scripts/start_scanner.py
```

Client and server output is saved in `logs/`, which is excluded from Git along
with exports, meshes, virtual environments, and local environment files.
The client reports missing hardware and retries automatically when no camera
is detected.

The **RGB**, **Depth**, and **Scanner** tabs display live camera views. The
Scanner tab shows RGB and depth side by side. To reconstruct a model, click
**Start Scan**, capture overlapping frames manually or with **Auto every**,
The **Live fused surface feedback** setting adds a persistent 3D view while
frames are processed during capture. Drag to orbit, scroll to zoom, and
double-click to switch color/shape. Pending frames and processing time show when
the server falls behind. Click **Preview Scan** for a full snapshot in a separate
viewer, or **Stop & Build Mesh** for final export.

To check the pipeline without connecting or using a Kinect:

```bash
OMP_NUM_THREADS=4 python scripts/check_scanner.py
```

This starts an isolated server on a free loopback port and uses synthetic frames
to check single/batch upload, ICP alignment, TSDF reconstruction, WebSocket
progress, preview, incremental mesh building, and readable PLY/OBJ exports.
It also checks the Qt client workflow with camera acquisition replaced by
synthetic input and an offscreen Qt platform; it does not open a 3D viewer.

---

## Reconstruction Quality and Public Replay

Configure near/far clipping, voxel size, final surface confidence, optional central
crop, and registered-RGB calibration **before starting a scan**. These settings
now affect reconstruction. Capture fresh overlapping views while moving the
Kinect around a stationary subject. “Save local RGB-D recording” enables later
replay without hardware; recordings stay outside Git.

The updated tracker reduced camera-position RMSE from about 220 mm to 44 mm on
an 80-frame public TUM Kinect replay using identical intrinsics (80/80 frames
accepted). Processing is slower; individual-device quality still needs a live
comparison. See [changes, benchmark details, reproduction commands, and remaining
plan](docs/SCAN_QUALITY.md).

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 python scripts/check_scanner.py --public-data
```

---

## Live Feedback, NVIDIA Compute, and Textures

New scans enable live feedback in the GUI; API clients opt in with
`live_reconstruction: true` in reset settings. CPU remains supported on Apple
Silicon. An NVIDIA server can select CUDA fusion, extraction, and tensor ICP:

```bash
KINECT_DEVICE=cuda KINECT_TRACKING=tensor KINECT_BLOCK_COUNT=5000 python -m scanner_server
```

CUDA requires a compatible NVIDIA driver and CUDA-enabled Open3D build.
An explicit unavailable CUDA request fails at startup. `KINECT_DEVICE=auto`
selects CUDA when available and reports its CPU fallback through health/status.
Actual CUDA performance needs hardware validation; it has not been measured here.

Final exports now include **textured GLB** (one file) and **textured OBJ bundle**
(ZIP containing OBJ, MTL, PNG, and a texture report). The existing plain OBJ uses
a vertex-color extension whose support varies between readers; it has no UV
texture. PLY preserves vertex colors. Textured exports use a separately simplified
mesh, defaulting to 50,000 triangles, a 1024-pixel atlas, and up to 24 RGB views.
They preserve the full final mesh for PLY/plain OBJ export.

**Save full RGB-D session** downloads lossless images, calibration, settings,
estimated poses, diagnostics, and timings. Unzip it for recording replay.
**Final pose refinement** is experimental and off by default: it validates loop
constraints, optimizes a bounded keyframe graph, and reintegrates into a fresh
volume only after separate geometry samples improve. It can retain the original
trajectory when no reliable loop exists, and needs memory for two volumes.

See [operation, validation, limits, and remaining implementation work](docs/LIVE_RECONSTRUCTION.md)
and [papers and open-source integration roadmap](docs/RESEARCH_ROADMAP.md).

---

## Architecture

```
 LAPTOP (Client)                        SERVER
 ┌──────────────────────┐              ┌──────────────────────────┐
 │  Kinect v1 sensor     │              │  FastAPI + ScanEngine    │
 │  ─────────────────    │   HTTP/WS    │  ──────────────────────  │
 │  KinectWorker         │◄────────────►│  ICP (CPU/CUDA)          │
 │  PyQt6 GUI            │  port 8000   │  TSDF (CPU/CUDA)         │
 │  Live RGB + depth     │              │  Mesh extraction         │
 │  Frame capture        │              │  PLY/OBJ export          │
 └──────────────────────┘              └──────────────────────────┘
        shared/                                shared/
     (config, protocol)                    (config, protocol)
```

The **client** captures Kinect frames and displays live video. Frames are compressed (zlib) and sent to the **server** over HTTP in batches. The server runs ICP registration + TSDF volumetric integration using Open3D (VoxelBlockGrid on the selected CPU/CUDA device), then serves the reconstructed mesh back to the client for preview and export.

| Component | Runs on | Key dependencies |
|-----------|---------|-----------------|
| **Client** (`kinect_scanner`) | Laptop with Kinect | PyQt6, freenect, httpx, websocket-client |
| **Server** (`scanner_server`) | Any machine with enough RAM | FastAPI, Open3D, uvicorn |
| **Shared** (`shared`) | Both | Pure Python (numpy only) |

---

## Scanning Workflow

```
Connect to Server ──> Start Scan ──> Capture Frames ──> Preview ──> Build Mesh ──> Export
                                      (batched to server)  (optional)  (on server)   PLY/OBJ
```

1. Enter the server IP and click **Connect**
2. Click **Start Scan** to begin a new session
3. Move the Kinect around the object, clicking **Capture Frame** or enabling auto-capture — frames are compressed, batched, and uploaded to the server
4. Optionally click **Preview Scan** to process frames and view the current mesh
5. Click **Stop & Build Mesh** to process all remaining frames and extract the final mesh on the server
6. **Export** as PLY or OBJ — the file is downloaded from the server and saved locally

---

## Server Setup

### Requirements
- Python 3.10+
- Open3D 0.19+

### Install

```bash
pip install -r requirements-server.txt
```

### Run

```bash
python -m scanner_server
```

The server listens on `0.0.0.0:8000` by default.

For standalone processes, `KINECT_SERVER_HOST` and `KINECT_SERVER_PORT`
configure the server's bind address and the client's connection fields.
`KINECT_AUTOCONNECT=1` makes the client connect on startup. The standalone
server retains a 50,000-block TSDF default; use `KINECT_BLOCK_COUNT` to change
it. The combined launcher always binds to `127.0.0.1`.

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/health` | Connection check + status |
| `GET` | `/api/scan/status` | Stored/integrated counts, has_mesh |
| `POST` | `/api/scan/reset` | Validate optional settings JSON, reset engine |
| `GET` | `/api/scan/diagnostics` | Accepted poses and per-frame quality/rejections |
| `POST` | `/api/scan/frame` | Upload a single compressed frame (binary body) |
| `POST` | `/api/scan/frames` | Upload a batch of compressed frames (binary body) |
| `POST` | `/api/scan/build` | Process all frames + build mesh |
| `POST` | `/api/scan/preview` | Process + extract preview, return PLY |
| `GET` | `/api/scan/export/ply` | Download mesh as PLY |
| `GET` | `/api/scan/export/obj` | Download mesh as vertex-colored OBJ |
| `GET` | `/api/scan/export/glb` | Download UV-textured GLB |
| `GET` | `/api/scan/export/obj.zip` | Download OBJ/MTL/PNG texture bundle |
| `GET` | `/api/scan/export/session` | Download lossless RGB-D session ZIP |
| `WebSocket` | `/ws/progress` | Build/preview progress and bounded live geometry |

---

## Client Setup (Laptop with Kinect)

### Requirements
- Xbox 360 Kinect (model 1414 / Kinect v1) with USB + power adapter
- libfreenect driver installed
- Same LAN as the server

### Install

```bash
pip install -r requirements-client.txt
```

#### Install libfreenect (Kinect driver)

```bash
git clone https://github.com/OpenKinect/libfreenect
cd libfreenect && mkdir build && cd build
cmake .. -DBUILD_PYTHON3=ON -DCMAKE_INSTALL_PREFIX=/usr
make && sudo make install
cd ../wrappers/python && pip install .
```

> See [docs/KINECT_V1_LINUX_PYTHON_REFERENCE.md](docs/KINECT_V1_LINUX_PYTHON_REFERENCE.md) for detailed installation, udev rules, and troubleshooting.

### Run

```bash
python -m kinect_scanner
```

### View Modes

| Mode | Description |
|------|-------------|
| **RGB** | Live color camera feed |
| **Depth** | Colorized depth map with adjustable near/far clipping |
| **Scanner** | Side-by-side RGB + depth for scanning |

---

## Network Protocol

Frames are serialized using the `shared.protocol` module.

### Single Frame

| Field | Size | Description |
|-------|------|-------------|
| `rgb_len` | 4 bytes (big-endian uint32) | Length of compressed RGB data |
| `rgb_compressed` | variable | zlib level=1 compressed RGB (480x640x3 uint8) |
| `depth_compressed` | remainder | zlib level=1 compressed depth (480x640 uint16) |

Typical compressed frame size: ~150-300 KB (vs ~1.5 MB uncompressed).

### Batch (multiple frames)

| Field | Size | Description |
|-------|------|-------------|
| `magic` | 4 bytes | `0x42415448` ("BATH") — identifies a batch payload |
| `frame_count` | 4 bytes (big-endian uint32) | Number of frames N |
| `lengths` | 4*N bytes | Per-frame packed byte lengths |
| `frames` | variable | Concatenated single-frame payloads |

The client batches consecutive captures up to 100 frames per request (8 in live mode). It preserves
reset/preview/build command barriers. New frames prepend `RGB2`, a 4-byte JSON
metadata length, and metadata before the original single-frame payload. Depth
bytes are little-endian millimetres. The server still accepts legacy packets;
new clients and servers should be upgraded together for metadata support.

---

## Scan Parameters

| Parameter | Value |
|-----------|-------|
| Voxel size | 5 mm |
| SDF truncation | 40 mm |
| Max depth | 4.0 m |
| Depth range | 500 - 4000 mm |

---

## Export Formats

| Format | Contents | Use case |
|--------|----------|----------|
| **PLY** | Point cloud or triangle mesh with vertex colors | MeshLab, CloudCompare, Blender |
| **OBJ** | Triangle mesh with a vertex-color extension | Geometry export; color support varies |
| **GLB** | UV mesh, embedded texture and material | Portable colored 3D asset |
| **Textured OBJ ZIP** | OBJ, MTL, PNG, texture report | Applications supporting OBJ materials |
| **Session ZIP** | Lossless RGB/depth, settings, estimated poses, diagnostics | Offline reconstruction and comparison |

---

## Project Structure

```
Kinect-3D-Scanner/
├── shared/                        # Shared pure-Python utilities
│   ├── config.py                  # Camera intrinsics, scan presets
│   └── protocol.py                # Frame pack/unpack (single + batch)
│
├── kinect_scanner/                # CLIENT (runs on laptop)
│   ├── __main__.py                # Entry point: python -m kinect_scanner
│   ├── config.py                  # Re-exports shared.config + O3D_INTRINSIC
│   ├── worker.py                  # Background Kinect frame capture thread
│   ├── server_client.py           # HTTP + WebSocket client
│   ├── server_task_worker.py      # Background task queue for server calls
│   ├── viewer.py                  # 3D mesh/point cloud visualization
│   ├── engine.py                  # Local scan engine (kept for reference)
│   ├── task_manager.py            # Local task manager (kept for reference)
│   └── gui/
│       ├── main_window.py         # PyQt6 application window
│       └── widgets.py             # Depth colorization, image conversion
│
├── scanner_server/                # SERVER (runs on processing machine)
│   ├── __main__.py                # Entry point: python -m scanner_server
│   ├── app.py                     # FastAPI app with all endpoints
│   ├── config.py                  # Server-side O3D_INTRINSIC
│   └── engine.py                  # Scan pipeline (ICP, TSDF, mesh)
│
├── docs/
│   └── KINECT_V1_LINUX_PYTHON_REFERENCE.md
├── export/                        # PLY/OBJ exports (client-side)
├── mesh/                          # Auto-saved scan meshes (client-side)
├── requirements.txt               # All dependencies (client + server)
├── requirements-client.txt        # Client-only dependencies
├── requirements-server.txt        # Server-only dependencies
└── README.md
```

---

## Documentation

- **[Kinect v1 Technical Reference](docs/KINECT_V1_LINUX_PYTHON_REFERENCE.md)** — Hardware specs, driver installation, Python API, camera intrinsics, calibration, point cloud generation, registration algorithms, mesh reconstruction, and export formats.

---

## License

This project is provided as-is for educational and personal use.
