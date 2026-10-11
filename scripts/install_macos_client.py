"""Install a checked Mac bundle only after its client and helpers have quit.

Uses macOS ditto/ps and the bundle's hardware-free --check. Copies into a
temporary sibling directory, retains the previous bundle for rollback, and
never closes clients, contacts a reconstruction server or resets a scan.
"""

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


EXECUTABLE = Path("Contents/MacOS/Kinect 3D Scanner")


def require_stopped(bundle):
    """Check original command paths, including helpers and renamed live bundles."""
    paths = {str(Path(bundle).absolute()), str(Path(bundle).resolve())}
    snapshot = subprocess.run(["/bin/ps", "-axww", "-o", "pid=,command="],
                              check=True, capture_output=True, text=True).stdout
    running = []
    for line in snapshot.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and any(fields[1].startswith(path + "/Contents/") for path in paths):
            running.append(fields[0])
    if running:
        raise RuntimeError(
            f"Cannot replace running bundle {bundle} (PIDs: {', '.join(running)}). "
            "Finish uploads, save unsaved work, then quit the client and its viewers. "
            "Keep them closed until installation finishes. The installed bundle was not changed."
        )


def executable_hash(bundle):
    digest = hashlib.sha256()
    with (bundle / EXECUTABLE).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_bundle(bundle):
    subprocess.run([str(bundle / EXECUTABLE), "--check"], cwd="/", check=True, timeout=60)


@contextmanager
def install_lock(destination):
    """Serialize installers; do not wait behind another bundle replacement."""
    import fcntl

    with (destination.parent / ".kinect-scanner-install.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Mac client installation is in progress") from exc
        yield


def install_bundle(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    destination_resolved = destination.resolve()
    if (source == destination_resolved or source in destination_resolved.parents
            or destination_resolved in source.parents):
        raise ValueError("Source and destination bundles must be separate")
    if destination.is_symlink():
        raise ValueError("Use a real destination bundle, not a symlink")
    if not (source / EXECUTABLE).is_file():
        raise ValueError(f"Missing Mac client executable: {source / EXECUTABLE}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with install_lock(destination):
        require_stopped(destination)
        checked_hash = executable_hash(source)
        staging_root = Path(tempfile.mkdtemp(prefix=".kinect-scanner-install-", dir=destination.parent))
        staged = staging_root / destination.name
        backup = staging_root / "previous.app"
        try:
            subprocess.run(["/usr/bin/ditto", str(source), str(staged)], check=True)
            subprocess.run(["/usr/bin/diff", "-qr", str(source), str(staged)], check=True)
            check_bundle(staged)
            if executable_hash(staged) != checked_hash:
                raise RuntimeError("Staged executable differs from the source build")
            # A client may have launched during the copy/check. Recheck before
            # moving any installed files, including for pre-guard client builds.
            require_stopped(destination)
            if destination.exists():
                destination.rename(backup)
            try:
                staged.rename(destination)
                check_bundle(destination)
                if executable_hash(destination) != checked_hash:
                    raise RuntimeError("Installed executable differs from the checked build")
            except BaseException:
                if destination.exists():
                    destination.rename(staged)
                if backup.exists():
                    backup.rename(destination)
                raise
        finally:
            # Keep the previous working bundle after success. On failure the
            # original has been restored and only disposable staging remains.
            if not backup.exists():
                shutil.rmtree(staging_root)
        print(f"Installed: {destination}\nChecked executable SHA-256: {checked_hash}")
        if backup.exists():
            print(f"Previous bundle retained: {backup}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Checked Kinect 3D Scanner.app build")
    parser.add_argument("--destination", type=Path,
                        default=Path.home() / "Applications/Kinect 3D Scanner.app",
                        help="Installed app path (default: ~/Applications/Kinect 3D Scanner.app)")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("Installation requires macOS")
    try:
        install_bundle(args.source, args.destination)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Installation failed: {exc}\n")


if __name__ == "__main__":
    main()
