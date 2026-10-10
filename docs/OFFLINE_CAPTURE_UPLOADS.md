# Offline capture and upload buffering

With live reconstruction disabled, each selected frame is saved in a temporary
lossless disk spool before capture is confirmed. Upload speed does not change
the requested minimum interval. A single upload worker drains consecutive
captures in batches of up to eight, keeping image memory bounded while the
backlog remains on disk. Build, inspection and project-save commands retain
their queue barriers; Finish uploads earlier captures before reconstruction.

The client shows captured frames including the pending spool, plus a pending
upload count. The offline capture sound confirms each local capture; server
acknowledgements do not repeat it. Live reconstruction keeps its existing
adaptive cadence and upload-confirmation sound.

The spool lives under the client data directory's `capture-queue/`. Each directory
has session settings in `session.json` and uncompressed NPZ files with `rgb`,
`depth` and JSON `metadata`. Only server-confirmed files are removed. Disk-write
failure pauses capture and reserves at least 1 GiB of free space. Failed or
uncertain uploads remain on disk, including after exit or starting another scan;
they block a build/save of that incomplete session. Recovery of those files is
currently manual; reopening the app does not automatically replay them.

Remote capture packets use RGB4: horizontal differences within each image row,
with uint16 depth differences separated into low/high byte planes before zlib
level 1 compression. The transform is reversible, including unsigned overflow.
No RGB/depth pixels, calibration or motion metadata are discarded. Loopback
uses the previous unpredicted, level 0 packets to avoid unnecessary CPU work.
The updated server reads both old packets and RGB4. The client sends RGB4 to
remote servers directly; update and restart the server before using this client.

## Measurement on the reported scan

Measured on macOS using the 29 selected 1280×1024 RGB / 640×480 depth captures
from 10 October 2026, 11:39–11:40 Sydney time. Each original and predicted packet
was decoded and compared against the original arrays. This is a local encoding
benchmark, not a measurement of improved network throughput.

| Measurement | Previous packets | Predicted packets |
| --- | ---: | ---: |
| Total payload, 29 frames | 89.88 MiB | 71.68 MiB |
| Median packing time per frame | 72.75 ms | 57.86 ms |
| Median decoding time per frame | 9.79 ms | 19.70 ms |

Payload decreased **20.25%**. Median local spool write time was **2.08 ms**, with
a maximum of **7.99 ms**. Spatial prediction trades additional decode work for
fewer transmitted bytes. Pixel equality was checked for every frame.

The original client log showed five-frame upload calls taking 3.1–4.6 seconds,
plus about 0.4 seconds of optional PNG recording. The 73 ms/frame encoder cannot
explain the full delay. New `Capture transport` logs report packing time,
payload bytes and request/acknowledgement time; `Frame ingestion` server logs
report receive, decode and storage time. These distinguish transfer throughput
from server ingestion costs on the next real scan. Optional local PNG recording
uses explicit fast lossless compression.

Verification includes 120 captures buffered while transport is blocked, Finish
ordering, partial/failed acknowledgements, disk exhaustion, both sensor
resolutions, unsigned depth extremes, lossless protocol round-trips, real
HTTP/WebSocket live and offline reconstruction, and the installed app's
hardware-free check of buffering, counters, sounds and Finish controls.
