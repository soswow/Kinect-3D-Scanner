# Offline capture and upload buffering

With live reconstruction disabled, each selected frame is saved in a temporary
lossless disk spool before capture is confirmed. Upload speed does not change
the selected minimum capture interval. Camera movement, uncertain visual
tracking and changes of tracking origin do not trigger extra snapshots.
Move slowly enough to keep overlapping views at the interval you choose, or
select a shorter interval. Continuous motion observations remain available
between selected captures for reconstruction at Finish.
A single upload worker drains consecutive
captures in batches of up to eight, keeping image memory bounded while the
backlog remains on disk. Build, inspection and project-save commands retain
their queue barriers; Finish uploads earlier captures before reconstruction.

Each capture carries all available intervening acceleration reads, including
failures, and compact intermediate visual observations. The bounded preview
history explicitly reports gaps. Finish additionally checkpoints and sends the
complete acceleration journal before requesting reconstruction, covering pauses
and long intervals without sending intermediate camera images. The server
exports this journal as `motion.json`; the optional full sensor archive remains
a separate research aid. Motion capability negotiation prevents silent loss
when a new client connects to an old server.
Each intermediate observation retains up to 192 spatially distributed measured
feature identities, pixels and depths; at most 80 feature-bearing observations
remain in the rolling window. Bounds and incomplete intervals are explicit.
The server jointly fits these observations with selected raw captures and then
checks the resulting maps against all selected depth views before fusion.

The client shows captured frames including the pending spool, plus a pending
upload count. The offline capture sound confirms each local capture; server
acknowledgements do not repeat it. Live reconstruction keeps its existing
adaptive cadence and upload-confirmation sound.

The spool lives under the client data directory's `capture-queue/`. Each directory
has session settings in `session.json` and uncompressed NPZ files with `rgb`,
`depth` and JSON `metadata`. Only server-confirmed files are removed. Disk-write
failure pauses capture and reserves at least 1 GiB of free space. Failed or
uncertain uploads remain on disk, including after exit or starting another scan;
they block a build/save of that incomplete session. After an upload fails, later
captures stay on disk until recovery so they cannot overtake the missing capture.

## Recover interrupted uploads

Choose **File → Recover Captures…** to inspect retained captures, including while
disconnected. Discovery does not upload, reset a scan or remove files. Reconnecting
checks the matching session's retained captures before enabling Finish and save.

- **Retry Matching Scan** resumes ordered uploads to the same idle server session.
  It requires captures made with receipt support on an updated client and server.
  A lost acknowledgement can be confirmed without adding a duplicate capture.
  The server rejects changed pixels/metadata, another session, or an insertion
  that would put a capture in the wrong position. Failure leaves the remaining
  files available for another retry or an offline recovery copy.
- **Save Recovery Project** writes a lossless project from that buffer's pending
  captures, with the original settings, calibration and metadata. This works
  without a server. It is a **partial capture set**: already-confirmed images
  that exist only on the server are not included, and it contains no finished
  mesh or invented camera poses. Open the copy and use Finish to reconstruct
  those retained observations. Saving a copy preserves the buffer and does not
  mark the current scan saved or change its project destination.

Older buffers can be exported when their capture order can be established, but
cannot be safely retried through the new acknowledgement protocol. Unreadable or
ambiguously ordered buffers are reported for manual recovery; the app does not
silently skip their files. Resetting/reopening a server project creates a new
session and cannot restore authority to retry an earlier session's captures.
The partial-recovery notice remains visible after opening the copy and is
preserved when saving or making a compact project from it.

New buffers use a versioned atomic session manifest, durable sequence filenames,
and per-capture receipts. A receipt binds the original native pixels and metadata
to the stream, sequence and expected server index. Only a matching acknowledgement
permits deletion. The server advertises `capture_protocol.capture_receipts: 1`;
receipt uploads opt in with `X-Capture-Receipt: 1` and `X-Scan-Session`. Older
clients retain their existing upload behavior. Updating a separately running
server requires pulling the changes and restarting it when its work is saved.

The server bounds streamed upload bodies before decoding: 6,000,000 bytes per
single packet and 600,000,408 bytes for the existing maximum 100-packet batch.
Oversized bodies receive HTTP 413, including uploads without an accurate
Content-Length. One capture request at a time may read/decode/wait for storage;
later requests wait before reading their bodies, so a long Finish cannot collect
an unbounded in-memory upload backlog. Reading a slow upload does not hold the
scan-state lock. These are ingress memory bounds, not changes to capture quality
or the supported packet formats.

Queued captures retain their originating session across reconnects. Current
clients send `X-Scan-Session` with single and batch uploads; the server checks
it while holding the scan lock and rejects stale uploads with HTTP 409 before
storing any of them. Unconfirmed spool files remain on disk. The server advertises
`capture_protocol.session_guard: 1` in health responses. Headerless older clients
remain supported, but require updating for protection against uploads crossing a
reset. Older servers ignore the header and must also be updated and restarted.
If reset or Open Project loses its response, capture and save/build/export actions
wait for an authoritative, settled server status. Reconnecting the same session
preserves its pending spool and failed-upload barrier.

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

**Also keep selected captures locally** keeps an acknowledged lossless PNG
recording separate from the temporary upload spool. Save Project and Blender export reuse matching
local capture pairs, downloading only the server's reconstruction/mesh, retained
server-only observations and any images unavailable locally. Matching uses the
current session, acknowledged server indices and native pixel hashes. The
[project save guide](LIVE_RECONSTRUCTION.md#full-session-preservation) describes
compression, optional JPEG photos and failure handling. This requires updating
and restarting the reconstruction server; an older server sends the full PNG
project. Confirmed upload-spool files are removed as before and are not a cache
for project saving.

Verification includes 120 captures buffered while transport is blocked, Finish
ordering, partial/failed acknowledgements, disk exhaustion, both sensor
resolutions, unsigned depth extremes, lossless protocol round-trips, real
HTTP/WebSocket live and offline reconstruction, and the installed app's
hardware-free check of buffering, counters, sounds and Finish controls.
