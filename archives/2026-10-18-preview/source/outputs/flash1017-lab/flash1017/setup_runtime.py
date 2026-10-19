"""Build an isolated runtime from the installed oMLX, validating source hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP = Path("/Applications/oMLX.app/Contents/Resources")


def main():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    expected = provenance["baseline_files_sha256"]
    source = APP / "omlx"
    for relative, digest in expected.items():
        path = source / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Installed oMLX source changed: {relative}; review and rebase patches first")
    patch_paths = sorted((ROOT / "patches").glob("*.patch"))
    identity = hashlib.sha256(b"".join(p.read_bytes() for p in patch_paths)).hexdigest()
    destination = ROOT / "_runtime"
    stamp = destination / "flash1017-build.json"
    if destination.exists():
        if stamp.exists() and json.loads(stamp.read_text())["patches_sha256"] == identity:
            print("Isolated runtime already prepared.")
            return
        raise FileExistsError("An existing isolated runtime needs review; it was not overwritten")
    shutil.copytree(source, destination / "omlx", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    try:
        for path in patch_paths:
            subprocess.run(["/usr/bin/patch", "-p1", "--batch", "-i", str(path)],
                           cwd=destination, check=True, capture_output=True, text=True)
        stamp.write_text(json.dumps({"patches_sha256": identity,
                                     "baseline_files_sha256": expected}, indent=2) + "\n")
    except BaseException:
        # Keep failed preparation inspectable; never touch installed app files.
        raise
    print(f"Prepared isolated runtime: {destination}")


if __name__ == "__main__":
    main()
