"""Portable lossless session export, without client-only OpenCV dependencies."""

import io
import json
import zipfile

from PIL import Image


def export_session(engine, path):
    manifest = {
        "version": 1,
        "depth_unit": "millimetres",
        "settings": engine.settings.to_dict(),
        "frames": [],
        "reconstruction": "reconstruction.json",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, ((rgb, depth), metadata) in enumerate(
            zip(engine.raw_frames, engine.frame_metadata)
        ):
            paths = {"rgb": f"rgb/{index:06d}.png", "depth": f"depth/{index:06d}.png"}
            for name, array in (("rgb", rgb), ("depth", depth)):
                buffer = io.BytesIO()
                Image.fromarray(array).save(buffer, format="PNG")
                archive.writestr(paths[name], buffer.getvalue())
            manifest["frames"].append(
                {
                    **paths,
                    "timestamp_s": metadata.get("timestamp_s", index),
                    "metadata": {**metadata, "server_index": index},
                }
            )
        archive.writestr(
            "manifest.json", json.dumps(manifest, indent=2, allow_nan=False)
        )
        archive.writestr(
            "reconstruction.json",
            json.dumps(engine.reconstruction_report(), indent=2, allow_nan=False),
        )
