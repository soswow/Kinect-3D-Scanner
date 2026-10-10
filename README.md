<p align="center">
  <h1 align="center">Kinect 3D Scanner</h1>
  <p align="center">
    A client/server application for 3D scanning with the Xbox 360 Kinect (v1).<br/>
    Capture frames on a laptop, process on a server, export to Blender-compatible PLY/OBJ.
  </p>
</p>

<p align="center">
  <a href="#server-setup"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows%20server-blue" alt="Linux, macOS, and Windows server"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.10+-yellow?logo=python&logoColor=white" alt="Python"></a>
  <a href="#usage"><img src="https://img.shields.io/badge/UI-PyQt6-green?logo=qt&logoColor=white" alt="PyQt6"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/server-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="docs/KINECT_V1_LINUX_PYTHON_REFERENCE.md"><img src="https://img.shields.io/badge/docs-reference-orange?logo=readthedocs&logoColor=white" alt="Docs"></a>
</p>

---

## Start the Client and Server Separately

The client and server have independent entry points and lifetimes. Closing the
client does not stop the server. Current launch support is a macOS desktop client
and a manually started server on macOS or Windows. They can run on separate LAN
machines or on the same Mac; use `localhost` in the client for a local server.

For daily use on Mac, open **Kinect 3D Scanner.app** from Finder, Dock, or
Applications. The app includes Python, Qt, Open3D, the freenect bindings and their
USB libraries; it does not search for a virtual environment or start a server.
It remembers the server address and port. Start the server independently, then
click **Connect** in the client. Only one application should use the Kinect at a time.

To build the client app, use an environment with working Kinect bindings and a
full Xcode 26+ installation selected by `xcode-select` (including Icon Composer):

```bash
python -m pip install -r requirements-packaging.txt
python scripts/build_macos_client.py
```

The build creates `dist/Kinect 3D Scanner.app` and runs a hardware-free startup
check that exercises the real Qt window, synthetic capture in a spawned process,
calibration/sound resources, preferences, shutdown, and the mesh helper process.
Copy the entire `.app` to Applications. Rebuild after changing the source code.
The icon source is `assets/icons/kinect-scanner-client.icon`, editable in Icon
Composer. The build compiles it into `Assets.car` for native Tahoe rendering and
generates a flattened PNG, all standard/Retina PNG sizes (16–1024 pixels), and an
ICNS fallback for older macOS versions. The bundle includes the catalog and ICNS
in `Contents/Resources`, with `CFBundleIconName` and `CFBundleIconFile` pointing
to the custom icon. Shipping the native catalog avoids Tahoe's extra backing
tile around legacy icons. To regenerate only the icon assets, run
`python scripts/build_macos_icon.py` on macOS.
The bundle targets the Mac architecture used to build it. It is signed locally
for local use; distributing it to other users requires Developer ID signing and
notarization. No Linux or Windows client bundles are provided yet.

Client logs rotate in `~/Library/Logs/Kinect3DScanner/client.log`. The bundled
client saves local recordings and preview meshes under
`~/Library/Application Support/Kinect3DScanner/`; the default export folder is
`~/Documents/Kinect 3D Scanner/export/`. Source runs retain the repository's
`recordings/`, `mesh/`, and `export/` folders.

Operational logs use UTC timestamps (the `Z` suffix), process/thread IDs and
server session IDs. `client.log` records scan commands, upload acknowledgements,
build stages received from the server, downloads, saves and error tracebacks.
`viewer.log` records mesh loading in the separate preview process; `capture.log`
records sensor-journal starts, stops and disk errors. Each log keeps a 5 MB active
file and three rotated backups. Preserve all these files soon after a failure.
Every 15 seconds, the client and viewer log resident memory, child-process memory,
available system memory, swap usage and free space on the log volume. This sampler
runs outside the UI thread, so it can report while the window is unresponsive.
The client also logs capture state, upload queue length and how many preview
frames were replaced because the UI fell behind. Preview delivery retains only
the latest camera frame; full sensor recording remains independent.

The server writes `logs/server.log` relative to its launch directory, with the
same rotation and resource sampling. Set `KINECT_SERVER_LOG` to choose another
path. Server logs include build progress and operation durations/failures. Update
and restart the server separately to enable new server logging; updating the Mac
app does not update a remote server. Logging cannot record events after a forced
kill or while the OS has suspended the process, and a full disk can prevent writes.

**Finish Scan is not Save Project.** It stops camera delivery immediately,
checkpoints the local sensor journal and shuts down camera acquisition (including
USB retries). It builds on the server, then downloads a mesh and opens a local
viewer. The camera stays off after success or failure, during inspection/export,
and when reconnecting to a finished scan. **Resume Capture**, **New Scan**, or
**Reset Scan** turns it back on. Reset clears the scan and opens camera setup
without capturing; a new scan waits for a fresh frame before resetting the server.
Preview downloads stream to disk and stop if they
would leave less than 256 MiB free. If the client crashes, reconnect to the same
still-running server and use **Save Project** before starting/resetting a scan.
The server retains its current scan in memory, not as an automatic disk project;
a server restart loses unsaved captures. Optional local selected-frame recordings
can preserve images, but the default accelerometer journals alone cannot restore
a scan. Use **Save Project** for a reopenable archive of images and the model.

For development, install the commands into your configured environment:

```bash
python -m pip install --no-deps -e .
kinect-scanner-server
# In a separate terminal, using the client environment:
kinect-scanner
```

The module commands `python -m scanner_server` and `python -m kinect_scanner`
also remain available. Installation/build are explicit steps; startup never
installs packages or compiles extensions. See the separate setup sections below
for each component's dependencies.

The software has been checked on Apple Silicon macOS with Python 3.13 and
Open3D 0.20. `requirements-macos-lock.txt` records the tested Mac dependencies;
`freenect` must be compiled separately before building the app. Synthetic checks
cover reconstruction and the client workflow; live capture and mesh building
have also been exercised on Kinect v1 hardware. Reconstruction quality depends
on overlap, camera calibration, and the subject's geometry.

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
The **Logs** view after Depth shows timestamped connection, capture,
reconstruction, save and warning events while capture continues. Messages appear
when a state or reason changes; unchanged retries stay quiet, and a failure
can appear again after recovery. Routine resource samples and detailed timings
stay in the diagnostic files. It keeps
the latest 2,000 lines; **Copy Logs** copies the visible history and **Clear Logs**
clears it without changing the log files on disk. Scan state, progress and
actionable viewer warnings remain visible. Hover over settings and their labels
for help, or the **Fused point cloud** title for the preview surface explanation.
Choose **Automatic** or **Manual** before **Start Scan**.
Automatic capture starts when the server acknowledges the new session; Manual
offers **Capture Frame**. Keep the subject stationary and move the Kinect slowly
around it with overlapping views. Turntable scanning is not supported.

With **Show live reconstruction** off, captures are retained and server-side
reconstruction/tracking checks wait until **Finish Scan**. **Color-assisted
tracking** still measures motion between camera frames and saves motion seeds
with selected captures. Those seeds help later registration; they do not prove
alignment. Enable live reconstruction for immediate reconstruction-loss alerts.
Without live reconstruction, an unverified camera-motion notice still prompts
slower movement and overlapping views; it does not stop capture.

Open **AprilTag tracking** in scan settings and check **Use AprilTags to assist
tracking** to add optional tag evidence. Select a dictionary and click **Add**;
use **Remove selected** to remove it. All listed dictionaries are detected
together in every camera frame during capture and every saved view processed by
the server, including Finish Scan. The choices are OpenCV's `DICT_APRILTAG_16h5`,
`DICT_APRILTAG_25h9`, `DICT_APRILTAG_36h10`, and `DICT_APRILTAG_36h11` ([OpenCV
dictionary documentation](https://docs.opencv.org/4.8.0/de/d67/group__objdetect__aruco.html)).
The checkbox starts off, with 36h11 selected by default; choices are remembered
and saved with sessions. Color-assisted tracking works without tags, and tags
can also be enabled independently of ordinary color features.

During scanning, the live color preview outlines detected tags and shows their
numeric IDs, without dictionary names. Green outlines mean the tag has usable
measured corners; amber outlines mean it was detected but cannot currently
provide tracking evidence. The preview uses the calibrated depth grid to align
the outlines and also works with **Show tracking flow** switched off. Overlays
are display-only: saved captures retain their original RGB pixels.

Keep labels stationary and use a unique ID for each physical label within its
dictionary. The same numeric ID in different dictionaries is supported. Repeated
IDs in one image and ambiguous cross-dictionary decodes are excluded. Printed
size is not required: matched corners use measured depth at both ends, on the
calibrated depth grid. Tags need clear corners, at least 20 pixels per side on
that grid, valid depth within the selected range/crop, and RGB/depth timing
within 20 ms. Detection uses exact codes rather than bit-error correction.
Tag motion must also pass independent depth-overlap checks; missing or rejected
tags leave ordinary tracking available. Both final registration modes can use
verified tag motion, including motion along otherwise ambiguous planes.
OpenCV 4.8 or newer with `cv2.aruco` is required when tags are enabled. Update
and restart the reconstruction server as well as the client to use these settings.

Selected captures are buffered losslessly on local disk and uploaded in the
background, so an upload backlog does not change the requested capture cadence.
The capture counter includes buffered frames and shows how many await upload.
**Finish Scan** waits for all selected captures to reach the server before
reconstruction. Insufficient disk space pauses capture; failed uploads retain
their local buffer files and stop a build or project save from claiming an
incomplete scan is complete. The buffer is separate from the optional permanent
local recording and reserves 1 GiB of free disk space.

Remote uploads use lossless spatial prediction before compression to reduce RGB
and depth payloads. Update and restart the reconstruction server together with
this client; there is no format negotiation. Client logs separate packing time,
request time and payload size; server logs separate receive, decode and storage
time. A localhost connection skips spatial prediction and compression work.

Scan actions stay visible while setup settings scroll independently. **Pause**
and **Resume Capture** retain the current scan; **Finish Scan** builds the final
model. **Cancel Scan** returns to setup without a build, offering to save or
discard unsaved captures. Click **Start Scan** afterward to restart.
**Reset Scan** also clears finished models and returns to live camera setup.
It offers to save unsaved work and waits for the server to confirm the reset;
capture stays off until you click **Start Scan**.
**Inspect Scan** generates a temporary mesh during capture and opens the
finished mesh after a build. Final inspection downloads the actual final mesh,
including any enabled final refinement, rather than an earlier preview.

The point cloud offers visible **Follow**, **Orbit**, **Color**, **Shape**, and
**Fit View** controls. In Orbit, left drag to rotate and scroll to zoom. Pan with
right drag (secondary-click and drag on a trackpad), middle drag, or Shift+left
drag. **Fit View** resets rotation, zoom, and pan. **Details**
shows diagnostic timings. Depth preview uses inclusive clipping bounds: black
means missing depth, gray means excluded depth, and a white outline marks the
crop. **Capture interval** sets the ordinary automatic cadence. Offline depth
scans retain extra overlapping views when camera movement grows or tracking
weakens. Live capture slows to match recent processing and upload/feedback times, with the
adjusted pace available by hovering over the interval. It allows one processing frame and one
waiting capture, including uploads, and waits if live feedback disconnects.
Move more slowly at longer intervals to preserve overlap between views.

For capture debugging, open **Tracking diagnostics** and enable **Show tracking
flow** during a scan with live reconstruction and color-assisted tracking.
The calibrated color preview shows verified/rejected feature motion, new
corners, and optional LK patch outlines. Counts, timing, and reference age
appear in the sidebar. See [capture flow diagnostics](docs/CONTINUOUS_VISUAL_TRACKING.md#capture-flow-diagnostics)
for the color legend and current tracking parameters.
Camera-side features now keep their identities while their measured support
remains reliable. Adaptive top-ups fill count and coverage gaps; the diagnostic
view reports tracks kept/added/retired and their lifetimes.

**Space** pauses/resumes capture and **C** captures in Manual mode, except while
editing fields. **Export…** selects textured GLB, textured OBJ ZIP, colored PLY,
or plain OBJ; texture choices appear in that dialog. **Inspect Scan** previews the current scan.
**Open Project…** reopens a saved ZIP, including older session ZIPs, without a Kinect connected.
**Save Project** (Cmd/Ctrl+S) keeps lossless captures, calibration, settings, poses, diagnostics,
sensor observations and the finished mesh. **File → Save Project As…** saves another copy.
Projects use the same ZIP format as sessions; the finished mesh is an additional optional member.
New Scan, Open Project, Cancel Scan, Reset Scan and close offer Save project / Discard / Cancel, and continue only
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
Kinect around a stationary subject. **Save Project** keeps selected RGB-D images
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

**Scan sounds** in the sidebar’s **Feedback** section plays a short confirmation
for each selected frame buffered on disk when live reconstruction is off. Upload
acknowledgements then stay silent. With live reconstruction on, the confirmation
plays when captured frames reach the server; a batch uses one cue. Both Automatic
and Manual modes use these cues. Click it to mute; the preference is remembered.
Rapid confirmations do not overlap. Skipped captures stay silent; live-mode
rejected or failed uploads stay silent. Tracking loss
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
Recorded-session CUDA measurements and optional fused confidence-weighted
fusion are documented in [CUDA performance](docs/CUDA_PERFORMANCE.md).
Further experiments targeting tracking, descriptor retrieval, model preparation
and recovery are documented in [CUDA pipeline experiments](docs/CUDA_EXPERIMENTS.md).
On Windows, start the measured hybrid recipe with
`./scripts/start_cuda_server.ps1 -Recipe hybrid`. It uses CPU pose tracking and
verification, fused confidence-weighted CUDA volume integration, exact CUDA feature matching,
keyframe preparation caching and lazy model preparation. The launcher detects
the existing virtual environment, starts a hidden server, checks health and
writes logs under `logs/`. Use `-Recipe baseline` for the CUDA tracking control;
see the experiment report for full scan timings and geometry checks.

Server starts, shutdown signals, Python exception tracebacks and exit summaries
are appended to `logs/server.lifecycle.log`. Native fatal errors also write
Python thread stacks to stderr (`logs/server.stderr.log` with the CUDA launcher),
including failures during interpreter teardown. Set `KINECT_LOG_DIR` to change
the lifecycle diagnostic directory.
The Windows CUDA launcher keeps a separate supervisor running after startup;
it records the child server's exit code even when the server is forcibly killed.
Other launch methods can use `python -m scanner_server.supervisor` with the same
server arguments. Windows native crash status codes are decoded where known.
An external kill does not always identify its cause or the program responsible.
If the entire process tree is killed, or the machine loses power, the supervisor
cannot write a final event either; the last start will have no matching exit.

The optional `-Recipe adaptive` tries verified ORB tracking before SIFT fallback.
Its measured fast-preview profile uses **Live voxel 10 mm / Final voxel 5 mm**
in the client's advanced scan settings. Finish still uses the original recorded
views at 5 mm; the live preview is coarser. The measured profile also uses a
**Final surface confidence 2**. Final volume storage is allocated automatically from the measured scan extent and available RAM/GPU memory. The historical benchmark used a 10,000-block cap. See the experiment report for the
CPU/CUDA comparison, retained views and restrictions. Adaptive fallback is
enabled only for that validated resolution pair; other settings use ORB.
Add `-CudaInput auto` to accelerate calibrated native RGB/depth preparation on
compatible CUDA installations. Compatibility probes check the installed CPU
implementation; unsupported inputs use CPU preparation and report the reason.
`-CudaInput on` requires CUDA preparation. The default is `off`.
The recommended measured fast-preview command is
`./scripts/start_cuda_server.ps1 -Recipe adaptive -CudaInput auto`, with the
10 mm live / 5 mm final client settings above. On the two original exports this
processed selected views about 3.0–3.5 times faster live and reduced combined
processing by about 40–44% versus the original CPU workflow. These gains include
the preview and tracking-policy changes; the matched-policy CUDA contribution
and per-view recovery latency are detailed in the experiment report.
Sensor confidence uses the original CPU calculation unless the separate
`-CudaConfidence auto` or `on` option is selected. That optional CUDA path uses
exact reduction order and compatibility probes; both flags default to `off`.
The fused path uses optional CuPy (`pip install -r requirements-cuda-fusion.txt`
for CUDA 12); `KINECT_CUDA_FUSION=auto` falls back to tensor integration when it
is unavailable, `tensor` selects the previous implementation, and `fused`
requires the optimized path.

Final exports now include **textured GLB** (one file) and **textured OBJ bundle**
(ZIP containing OBJ, MTL, PNG, and a texture report). The existing plain OBJ uses
a vertex-color extension whose support varies between readers; it has no UV
texture. PLY preserves vertex colors. Textured exports use a separately simplified
mesh, defaulting to 50,000 triangles, a 1024-pixel atlas, and up to 24 RGB views.
They preserve the full final mesh for PLY/plain OBJ export.
Both texture modes choose photos over connected surface regions, with a
depth-tested fallback where the region's photo cannot observe a texel. The
default softens compatible photo boundaries in a three-texel strip and keeps
the interior detail from one photo. **Keep photo boundaries sharp** disables
that strip. **Match photo brightness when reliable** independently estimates
bounded RGB gains from shared observations; it can leave photos unchanged when
held-out overlap does not improve consistently. The OBJ bundle's
`texture-report.json` records that decision, source selection, and seam coverage;
GLB embeds the same report in mesh metadata. The client Logs also say whether
brightness matching was applied or left the photos unchanged.

**Save Project** packages lossless selected images, calibration, settings,
estimated poses, diagnostics, the finished mesh when available, and the client's continuous accelerometer log,
including failed reads. Camera images are saved only for selected captures by
default. Live visual tracking can process intermediate images without retaining
them all on disk. Each capture sends the intervening acceleration reads and
compact visual observations. Finish automatically uploads the complete small
acceleration journal, including pauses and failed reads, before reconstruction;
the optional full image archive is unnecessary for that transport.
Normal archive size depends on the selected captures; the motion
log adds a small amount of text data. Opening restores saved poses by fresh fusion and keeps capture paused; a saved final mesh is immediately available for export. Independent sensor observations remain in the project when it is saved again.

Saving reports preparation, download progress in MB, disk completion and sensor merging.
Downloads stream into a temporary file and replace the destination only after completion.
Client logs record preparation/download/disk timings; server logs record image encoding.
A slow connection can take much longer to download an archive than it takes to encode it.

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
for stronger assistance. To record and fit one with step-by-step terminal
guidance, close the scanner and run in its Python environment:

```bash
python scripts/calibrate_accelerometer.py --interactive --output ~/Documents/measured-accelerometer.json
```

Follow the nine physical positioning prompts, then select the output JSON with
**Load Accelerometer Calibration…** before scanning. Keep the head tilt fixed;
independent camera-axis alignment checks are required for a verified profile.
Hardware orientation/performance and tracking gains
still need validation; see the [implementation and calibration guide](docs/KINECT_ACCELEROMETER.md).
**Final registration** offers **Depth, color and motion (experimental)** and
the existing fragment registration. New GUI scans start with live fusion off;
**Show live reconstruction** remains available and saved preferences are kept.
Depth mode considers every raw depth capture independently of accepted live
poses. Measured RGB-D matches can constrain textured planes and identify loop
closures; camera-side visual history proposes initial poses. When enabled,
reliable accelerometer gravity supplies a broad orientation constraint and a
soft search weight. It supplies neither yaw nor integrated position. Depth-only
reconstruction remains supported with color and gravity assistance disabled.
The mode permits single-pair bridges when measurements support them,
checks all available component views for contradictory empty space, and retains
unconnected components with their own camera poses. Read the
[depth-only registration checks and experiments](docs/FRAGMENT_RECONNECTION.md#experimental-depth-only-final-registration).
The [10 October fast-room replay](docs/FULL_ROOM_SCAN_20261010.md) documents the
ordinary-upload result and remaining runtime, coverage and validation limits.

In existing fragment mode, **Reconnect separated views at Finish** is enabled
for new GUI scans. Finish
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
for backlog. Texture exports preserve detail with connected photo regions and
optional narrow boundary softening, and can independently match photo brightness.
Experimental reconstruction quality options remain off until measured scans justify them.

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
6. Use **Export…** for a model or **Save Project…** for a reopenable ZIP. **Open Project…** accepts existing session recordings too.
7. **Final voxel size** controls reconstruction detail; the app measures scene extent and allocates volume storage automatically. Old session block counts are accepted for compatibility and do not limit reconstruction. If memory is insufficient, the build reports estimated/available RAM or GPU memory before fusion and retains the scan. Choose a coarser final voxel or free memory on the processing machine, then **Retry Build**.

---

## Server Setup

### Windows CUDA server (no Kinect required)

Install 64-bit Python **3.12** and Visual Studio 2022 or Build Tools 2022 with
the **Desktop development with C++** workload (MSVC and a Windows SDK).
Use a compatible NVIDIA GPU and driver. The CUDA wheel below is specific to
CPython 3.12 on Windows x64; it does not work with other Python versions.
The CUDA toolkit is unnecessary for this setup: the wheel supplies its runtime,
and the scanner's native extension uses the C++ compiler.

Open PowerShell in a checkout of this repository. Ensure `python --version`
reports `3.12.x`, then create a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-server-windows-cuda-lock.txt
.\.venv\Scripts\python.exe -m pip install ./native
$env:KINECT_DEVICE = "cuda"
$env:KINECT_NATIVE = "on"
$env:OMP_NUM_THREADS = "4"
.\.venv\Scripts\python.exe scripts/check_scanner.py --server-only --startup-timeout 180 --timeout 300
```

The [official Open3D 0.20 Windows CUDA wheel](https://github.com/isl-org/Open3D/releases/tag/v0.20.0)
is a preview release. Its NVIDIA runtime dependencies install into the environment;
the machine still needs a compatible NVIDIA driver. The ordinary Windows
`open3d` package is CPU-only: do not install it alongside `open3d-cuda` in this
environment, since both provide the same Python module. Kinect drivers, freenect,
and Qt are unnecessary on the server.

First-use CUDA driver compilation can take several minutes; the longer check
timeouts above allow that initialization. Later runs reuse the driver's cache.
`requirements-server-windows-cuda-lock.txt` records the dependency versions
used for Windows validation. `requirements-server-windows-cuda.txt` is the
unpinned alternative for testing newer dependencies.

The Windows CUDA preview also has an upstream [process shutdown issue](https://github.com/isl-org/Open3D/issues/6399):
reconstruction tests can finish successfully, then Python exits with
`CUDA runtime error: driver shutting down`. During Windows validation, the
server regression assertions and synthetic HTTP scan/exports passed, but the
regression process exited abnormally during CUDA teardown. Explicit cache cleanup did not
resolve the exit error. For a server that fails to exit during CUDA teardown,
stop its process through Windows Task Manager.

Install the server command after the dependencies and native extension:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\kinect-scanner-server.exe --device cuda --native on --tracking tensor --threads 8
```

Those explicit options require CUDA and native kernels, so missing components
fail visibly. Without options, startup selects available CPU/CUDA and optional
native kernels automatically. The shared startup defaults are port 8000, 5,000
voxel blocks, 500 stored frames and four OpenMP threads. CLI options override
existing environment variables. Press Ctrl+C to stop.

For a source-checkout shortcut, double-click **Start Server.cmd**. It uses only
that checkout's `.venv` and the same Python startup policy. The optional
`scripts/start_server.ps1` wrapper forwards the same arguments. Neither wrapper
starts the client or creates a hidden background process.

```powershell
.\.venv\Scripts\kinect-scanner-server.exe --device cuda --native on --port 8001
```

Check `http://127.0.0.1:8000/api/health` for `device: CUDA:0`, tensor tracking,
and active native kernels. On the capture laptop, enter this server's LAN IPv4
address and port 8000. Windows Firewall must allow inbound TCP 8000 for the
project's Python interpreter. A LAN-scoped rule can be added once in an
administrator PowerShell window:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/allow_server_firewall.ps1
```

For a custom port, pass the same value with `-Port 8001`. Use the server on a
trusted LAN: its HTTP/WebSocket API does not authenticate clients. The firewall
helper limits the allowed remote addresses to the local subnet, including when
Windows marks that LAN as a Public network.

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

### Run (macOS and Windows)

Install dependencies from the appropriate server requirements file, then install
this project's command. On macOS, a fresh environment can be set up with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-server.txt
.venv/bin/python -m pip install ./native
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/kinect-scanner-server
```

The server stays in the foreground and prints startup status and request logs.
Ctrl+C stops it. The installed command works from any directory; alternatively,
run `python -m scanner_server` from the checkout. On Windows use the commands in
the Windows setup section above. Standalone server executables are not bundled
in this version; the installed Python command is the supported launch path.

The server binds to `0.0.0.0:8000` by default, allowing LAN connections. Use
`--host 127.0.0.1` when only local connections are needed. The client uses a
concrete server hostname/IP, such as `localhost`, rather than the bind address.
macOS uses the CPU backend when CUDA is unavailable.

| CLI option | Environment fallback | Default |
|------------|----------------------|---------|
| `--host` | `KINECT_SERVER_HOST` | `0.0.0.0` |
| `--port` | `KINECT_SERVER_PORT` | `8000` |
| `--device` | `KINECT_DEVICE` | `auto` (`auto`, `cpu`, `cuda`) |
| `--tracking` | `KINECT_TRACKING` | `auto` (`auto`, `legacy`, `tensor`) |
| `--native` | `KINECT_NATIVE` | `auto` (`auto`, `off`, `on`) |
| `--threads` | `OMP_NUM_THREADS` | `4` |
| `--block-count` | `KINECT_BLOCK_COUNT` | `5000` |
| `--max-frames` | `KINECT_MAX_FRAMES` | `500` |

CLI options override environment variables. Invalid settings fail before the
server loads its reconstruction dependencies. Advanced backend settings remain
available through their existing environment variables. `scripts/start_cuda_server.ps1`
is a separate research tool for measured CUDA recipes, not the normal server entry point.

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
| `GET` | `/api/scan/export/session` | Download project ZIP with lossless RGB-D captures and optional final mesh |
| `POST` | `/api/scan/project` | Upload and open a project/session ZIP; invalid archives retain the active scan |
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

For daily use on macOS, open **Kinect 3D Scanner.app**; see the build steps at
the top of this README. For development, install `python -m pip install --no-deps -e .`
and run `kinect-scanner` in the client environment. Both launch only the client.
Enter the server's LAN hostname/IP and port in the GUI, then click **Connect**.
The client remembers these fields across launches. `KINECT_SERVER_HOST` and
`KINECT_SERVER_PORT` override the saved connection fields for source runs;
`KINECT_AUTOCONNECT=1` optionally connects on startup.

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

Metadata is bounded to 1 MiB per capture. `/api/health` advertises motion
protocol support; the client retains queued captures and asks for a server
update if the old server cannot receive it. `/api/scan/motion` receives a
session-bound acceleration journal up to 32 MiB. Project uploads support up to
64 GiB, with separate limits for manifests, images, meshes and sensor journals.

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

- **[Script and research index](scripts/README.md)** — Maintained tools, reusable benchmarks, active research and archived experiments, with prerequisites and supported commands.
- **[CUDA scanning performance and research](docs/CUDA_EXPERIMENTS.md)** — Measured field recipe, complete archive comparisons, CUDA research results and reproduction instructions.
- **[Saved CUDA reports](docs/benchmarks/CUDA_REPORTS.md)** — Published charts and summaries, with original and published file hashes.
- **[Kinect v1 Technical Reference](docs/KINECT_V1_LINUX_PYTHON_REFERENCE.md)** — Hardware specs, driver installation, Python API, camera intrinsics, calibration, point cloud generation, registration algorithms, mesh reconstruction, and export formats.
- **[Accelerometer, sensor sessions, and portrait capture](docs/KINECT_ACCELEROMETER.md)** — Implemented concurrent reads, conservative gravity assistance, full-stream archival/replay, calibration, and portrait previews/exports. Hardware validation remains pending.

---

## License

This project is provided as-is for educational and personal use.
