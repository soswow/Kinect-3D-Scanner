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
repository's `.venv/`, the parent folder's `.venv/`, an activated virtual
environment, or `python3` from PATH, in that order. Finder launches do not
require activating either local environment in Terminal first.
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
| `KINECT_NATIVE` | `auto` | Use installed C++ kernels; `off` selects NumPy, `on` requires native |

```bash
KINECT_SERVER_PORT=8001 KINECT_BLOCK_COUNT=10000 python scripts/start_scanner.py
```

Client and server output is saved in `logs/`, which is excluded from Git along
with exports, meshes, virtual environments, and local environment files.
The client reports missing hardware and retries automatically. Camera acquisition
runs in an isolated process: a stalled USB driver times out and restarts, and
cannot prevent the window from closing. RGB/depth pairing uses the Kinect v1's
60 MHz device clock, with wraparound handled before conversion to milliseconds.
Acquisition retains a short depth history and pairs each RGB frame with the
nearest unused depth timestamp, including RGB callbacks delivered late. These
driver timestamps mark packet completion, rather than measured exposure times.

With the scanner closed, check your connected camera and window shutdown:

```bash
python scripts/check_camera.py --window --offscreen
```

The check prints frame counts, preview status, pairing offset, and shutdown time;
it does not save camera images.

The **Scan**, **Color**, and **Depth** views show the reconstruction or a full
camera preview. In Scan view, the fused point cloud is prominent and the color
and depth camera previews are stacked beside it, with color above depth.
Choose **Automatic** or **Manual** before **Start Scan**.
Automatic capture starts when the server acknowledges the new session; Manual
offers **Capture Frame**. Keep the subject stationary and move the Kinect slowly
around it with overlapping views. Turntable scanning is not supported.

Scan actions stay visible while setup settings scroll independently. **Pause**
and **Resume Capture** retain the current scan; **Finish Scan** builds the final
model. **Cancel Scan** returns to setup without a build, offering to save or
discard unsaved captures. Click **Start Scan** afterward to restart.
**Inspect Scan** generates a temporary mesh during capture and opens the
finished mesh after a build. Final inspection downloads the actual final mesh,
including any enabled final refinement, rather than an earlier preview.

The point cloud offers visible **Follow**, **Orbit**, **Color**, **Shape**, and
**Fit View** controls. In Orbit, drag to rotate and scroll to zoom. **Details**
shows diagnostic timings. Depth preview uses inclusive clipping bounds: black
means missing depth, gray means excluded depth, and a white outline marks the
crop. **Minimum capture interval** sets the fastest automatic cadence. Live
capture slows to match recent processing and upload/feedback times, with the
adjusted pace shown below the interval. It allows one processing frame and one
waiting capture, including uploads, and waits if live feedback disconnects.
Move more slowly at longer intervals to preserve overlap between views.

For capture debugging, open **Tracking diagnostics** and enable **Show tracking
flow** during a scan with live reconstruction and color-assisted tracking.
The calibrated color preview shows verified/rejected feature motion, new
corners, and optional LK patch outlines. Counts, timing, and reference age
appear in the sidebar. See [capture flow diagnostics](docs/CONTINUOUS_VISUAL_TRACKING.md#capture-flow-diagnostics)
for the color legend and current tracking parameters.

**Space** pauses/resumes capture and **C** captures in Manual mode, except while
editing fields. **Export…** selects textured GLB, textured OBJ ZIP, colored PLY,
or plain OBJ; texture choices appear in that dialog. **Open Model…** opens a file.
**Save Session…** preserves lossless observations for replay. New Scan, Cancel Scan and close
offer Save Session / Discard / Cancel for unsaved captures, and continue only
after a requested save succeeds. Reconnecting restores an existing server scan
paused; a failed build offers retry or resumed capture without resetting frames.

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
crop, RGB capture mode, and calibration **before starting a scan**. These settings
now affect reconstruction. Capture fresh overlapping views while moving the
Kinect around a stationary subject. **Save Session** keeps selected RGB-D images
and a continuous accelerometer/timing log for later reanalysis. Full camera-stream
recording is optional and off by default. “Also keep selected captures locally”
creates a separate local recording too. Recordings stay outside Git.

The GUI now loads the [complete measured Kinect calibration](calibration/default.json)
by default: independent RGB profiles at both resolutions, IR/depth intrinsics
and distortion, IR–depth grid correspondence, IR-to-RGB pose, and board-checked
raw-depth conversion. RGB capture defaults to **1280 × 1024 at 10 fps**; select
**640 × 480 at 30 fps** in scan settings when preferred. Depth remains native
640 × 480 raw disparity. Reconstruction uses the calibrated depth grid, while
texture exports sample original full-resolution RGB. **Load Calibration…**
requires the complete publication. Single-camera JSON files are rejected. Recordings
retain original observations and all calibration data. See the
[JSON structure and runtime conventions](calibration/README.md).

**Automatic** follows fresh camera frames. Its interval rounds to whole frame
periods: 0.1-second steps in 10 fps mode, or 1/30-second steps in 30 fps mode.
A 0.5-second interval selects every five high-resolution pairs or fifteen VGA
pairs. Capture waits for fresh input and reconstruction capacity; delays extend
the interval without creating duplicate captures or catch-up bursts.

Expand **RGB camera** in scan setup to choose **Auto exposure** (default) or
**Manual exposure**. Manual accepts reciprocal seconds, such as **1/125 s** or
**1/250 s**: a larger denominator gives a faster shutter and less motion blur.
The slowest choice is 1/10 s at 10 fps or 1/30 s at 30 fps. Changing exposure
restarts the camera preview; wait for fresh frames before starting a scan.
Set exposure before capture, as with the other scan settings. The camera's
reported exposure time appears in the section; hardware quantizes the request.
The **Sensitivity (gain)** dropdown selects 1×, 2×, 4×, or 8× analog gain.
Higher gain brightens the image and increases noise. Both controls are remembered
and included in scan settings, recordings, and frame metadata.

Kinect v1's [libfreenect exposure API](https://github.com/OpenKinect/libfreenect/blob/master/include/libfreenect.h)
supports automatic exposure or a fixed shutter time. The sensor does not provide
an independent automatic-gain mode with a fixed shutter. Manual exposure uses
the MT9M112 sensor's gain registers; gain multipliers have no calibrated ISO
mapping. RGB white balance settles before manual mode freezes its colour ratios.
Manual then disables AE, AWB, and flicker, clears automatic digital gain and fine
shutter delay, and applies shutter and analog gain. Controls are checked again
after new frames arrive and periodically during capture. Camera reconnects reapply
them. See the manufacturer's [MT9M112 datasheet, Tables 15 and 16](https://dlscorp.com/wp-content/uploads/2019/03/MT9M112_DS_full.pdf).

The client uses the public C exposure functions from the same libfreenect library
already loaded by the Python extension when its Python bindings omit them. This
requires OpenKinect's `DevPtr` representation. Manual gain also requires exported
`read_cmos_register`/`write_cmos_register` functions (available in the tested macOS
bindings) or Python wrappers for them, and checks the sensor ID before writing.
Unsupported bindings retain automatic operation; manual requests fail visibly.
With the scanner closed, test a manual shutter and gain directly:

```bash
python scripts/check_camera.py --exposure manual --shutter-speed 250 --gain 2
```

**Scan sounds** in the toolbar plays a short confirmation when captured
frames reach the server, in Automatic and Manual modes. Click it to mute; the
preference is remembered. A batch of frames uses one cue, and rapid confirmations
do not overlap. Skipped, rejected or failed uploads stay silent. Tracking loss
plays a distinct descending double tone once per loss episode; verified recovery
plays its rising reverse once. Both take priority over capture confirmations,
which resume after the recovery tone finishes. Starting, cancelling, or restoring
another session stays silent. The same mute control applies to all scan sounds.

Your choices save automatically as you edit them and restore on the next launch:
capture mode and interval, clipping and crop, recording, reconstruction and
experimental options, camera resolution and exposure, shutter speed and gain, calibration,
server host/port, sound,
and the last accepted export format and texture options. Calibration is saved
as a complete snapshot, so its original JSON file can be moved afterward.
Preferences use Qt's per-user settings store. `KINECT_SERVER_HOST` and
`KINECT_SERVER_PORT` override the saved connection target for that launch.
Reconnecting to an existing server scan restores that scan's setup without
replacing your saved defaults. Capture starts only after **Start Scan** or
**Resume**, including when Automatic is your saved mode.

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

On tracking loss, fusion pauses until a recent raw view or earlier keyframe
verifies the camera position. Automatic capture selects a sharp recent pair
and probes again as soon as the previous recovery check finishes.
The red notice, overhead trajectory, and saved reference image guide recovery;
RGB/depth timing warnings explain when color recovery is unavailable. See
[tracking recovery](docs/TRACKING_RECOVERY.md) and the
[RGB/depth timing explanation and second chest-session analysis](docs/RGB_DEPTH_TIMING.md).
The [Chest 4 recovery results](docs/CHEST_4_TRACKING_RECOVERY.md) cover recent-view
fallback, sharp capture selection, returning-loop evidence, and replay limits.

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

**Save Session** packages lossless selected images, calibration, settings,
estimated poses, diagnostics, and the client's continuous accelerometer log,
including failed reads. Camera images are saved only for selected captures by
default. Live visual tracking can process intermediate images without retaining
them all on disk. Normal archive size depends on the selected captures; the motion
log adds a small amount of text data.

**Record all camera frames (large files)** is an explicit, off-by-default option
for research requiring exact intermediate-frame replay. Only this option adds
lossless NPY streams at about **3.5 GB/minute** for high-resolution RGB or
**2.8 GB/minute** for VGA. The UI reports dropped data
or recording errors, and the archive carries an explicit completeness report.
Recording pauses at the save boundary and resumes afterward; data before a
reconnect is kept in separate clock epochs. Existing local recordings remain
after saving. Unzip the session for replay:

```bash
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset recording --path /path/to/unzipped-session \
  --stride 1
```

For a session captured with all camera frames enabled, add
`--sensor-streams --recompute-motion --stride 5` to reconstruct intermediate motion.

The toolbar offers **Auto / Landscape / Portrait left / Portrait right**. Auto
uses filtered acceleration with hysteresis; manual locks work without a sensor.
RGB and depth previews rotate together. Selected portrait captures include
ordinary rotated PNG copies under `display/`, alongside native images and
calibration. Auto orientation is the default and works independently of the
off-by-default **Use accelerometer to assist tracking** checkbox in scan settings.
Enabling this checkbox adds a bounded roll/pitch correction
to the existing RGB-D initializer and retains the normal acceptance checks.
The factory sensor axes are provisional; load a measured accelerometer profile
for stronger assistance. Hardware orientation/performance and tracking gains
still need validation; see the [implementation and calibration guide](docs/KINECT_ACCELEROMETER.md).
**Reconnect separated views at Finish** is enabled for new GUI scans. Finish
reconstructs local fragments from retained raw frames, verifies overlapping
fragments, optimizes their pose graph, and rebuilds a fresh volume from the
connected views. Unconnected fragments remain in the saved session and are
reported explicitly. This runs offline at Finish and may leave a partial model
when there is insufficient overlap or repeated geometry. For an older ZIP, run:

```bash
python scripts/reconnect_session.py export/your-session.zip \
  --output-dir export/reconnected --save-session
```

See [fragment reconnection and its limits](docs/FRAGMENT_RECONNECTION.md).
The [algorithm review and recorded-session results](docs/ALGORITHM_REVIEW.md)
describe the tracking, reconstruction, and performance changes measured on
three saved scans, including their remaining limitations.

**Final pose refinement** is experimental and off by default: it validates loop
constraints, optimizes a bounded keyframe graph, and reintegrates into a fresh
volume only after separate geometry samples improve. It can retain the original
trajectory when no reliable loop exists, and needs memory for two volumes.

**Joint RGB-D refinement at Finish** is also experimental and off by default.
It builds features shared by at least three saved views, then jointly refines
camera positions and 3D feature positions using color and measured depth.
Disjoint depth samples from every output view must improve before a bounded
fresh reconstruction replaces the previous volume. Calibration and the first
camera stay fixed. Current limits are 24 keyframes, 800 features, 128 accepted
views and a 45-second proposal budget. Install the updated server requirements
for SciPy. See [joint refinement](docs/JOINT_RGBD_REFINEMENT.md) and
[candidates, statistical interpretation and evaluation](docs/RGBD_IMPROVEMENT_PLAN.md).

Optional **Lost tracking recovery**, **sensor confidence weighting**, and
**finer final fusion** now have explicit controls. Final fusion uses a separate,
bounded volume; live feedback reports queue age and can pause automatic capture
for backlog. Texture exports can match exposures or select one best view per
texel. Experimental quality options remain off until measured scans justify them.

See [tested milestone checkpoints and next evidence](docs/IMPLEMENTATION_MILESTONES.md),
[operation, validation, limits, and remaining implementation work](docs/LIVE_RECONSTRUCTION.md)
[capture performance measurements and reproduction](docs/CAPTURE_PERFORMANCE.md),
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
Ready ──> Start Scan ──> Capture ↔ Pause ──> Finish Scan ──> Inspect / Export
```

1. Use automatic local connection, or expand **Connection details** to connect remotely.
2. Wait for fresh camera frames, set the scan range, and choose Automatic or Manual.
3. Click **Start Scan** and move the Kinect around the stationary subject. In Manual, use **Capture Frame**.
4. Pause/resume as needed, or **Cancel Scan** to return to setup without building. **Inspect Scan** temporarily suspends capture to prepare a mesh snapshot.
5. Click **Finish Scan**. The finished mesh opens for inspection; failed builds retain captures for retry/resume.
6. Use **Export…** for a model or **Save Session…** for replayable source captures.

---

## Server Setup

### Requirements
- Python 3.10+
- Open3D 0.19+

### Install

```bash
pip install -r requirements-server.txt
```

For faster calibrated RGB-D preparation and CPU confidence-weighted fusion,
install the optional C++ extension using the same Python environment as the
server (requires a C++17 compiler and Python development headers):

```bash
python -m pip install ./native
```

`KINECT_NATIVE=auto` uses a compatible installed extension and otherwise keeps
the NumPy implementation. `KINECT_NATIVE=on` fails at startup if the extension
is unavailable, making performance comparisons explicit; `off` selects the
reference implementation. Health/status and exported reconstruction reports
include `backend.native_kernels`. Reinstall `./native` after editing its C++
sources or changing Python environments. Installation builds the extension;
the scanner never compiles it during startup.

The C++ kernels preserve clipping, filtering, RGB projection/occlusion, and
confidence-weighted voxel-update rules. OpenCV still samples color, and Open3D
still performs tracking, volume allocation, extraction, and CUDA integration.
See [native backend measurements](docs/NATIVE_PERFORMANCE.md) for recorded-session
comparisons and reproduction commands.

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
| `rgb_compressed` | variable | Lossless zlib RGB (480x640x3 or 1024x1280x3 uint8); level 0 on loopback, level 1 remotely |
| `depth_compressed` | remainder | zlib level=1 compressed depth (480x640 uint16; session defines units) |

High-resolution frames prepend `RGB3`, a big-endian uint32 JSON length, and a
JSON header containing `rgb_shape: [1024, 1280, 3]`, followed by this payload.
VGA metadata packets use `RGB2`; original packets remain supported. Session
settings define whether depth contains raw 11-bit disparity or registered
millimetres. See [calibration conventions](calibration/README.md).


Uncompressed native high-resolution RGB plus depth is about 4.5 MB per frame;
compression depends on image detail.

### Batch (multiple frames)

| Field | Size | Description |
|-------|------|-------------|
| `magic` | 4 bytes | `0x42415448` ("BATH") — identifies a batch payload |
| `frame_count` | 4 bytes (big-endian uint32) | Number of frames N |
| `lengths` | 4*N bytes | Per-frame packed byte lengths |
| `frames` | variable | Concatenated single-frame payloads |

The client batches consecutive captures up to 100 frames per request (8 in live mode). It preserves
reset/preview/build command barriers. Frames prepend `RGB3` for high-resolution
RGB or `RGB2` for VGA, a 4-byte JSON metadata length, and metadata before the
single-frame payload. Depth bytes are little-endian uint16 raw disparity for
Kinect sessions; explicit public dataset replay settings use millimetres.

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
- **[Accelerometer, sensor sessions, and portrait capture](docs/KINECT_ACCELEROMETER.md)** — Implemented concurrent reads, conservative gravity assistance, full-stream archival/replay, calibration, and portrait previews/exports. Hardware validation remains pending.

---

## License

This project is provided as-is for educational and personal use.
