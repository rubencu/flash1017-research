# SPDX-License-Identifier: Apache-2.0
"""Verify retained historical bytes; no model, imports of runtime code or server."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / "ARCHIVE-INDEX.json").read_text())
    if index.get("schema") != "flash1017-research-archives/1":
        raise ValueError("Unsupported archive index")
    checked, checksums = 0, []
    for snapshot in index["snapshots"]:
        relative = PurePosixPath(snapshot["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe archive path")
        archive = root / relative
        archive.resolve().relative_to(root)
        expected = set()
        for item in snapshot["files"]:
            name = PurePosixPath(item["path"])
            if name.is_absolute() or ".." in name.parts or item["path"] in expected:
                raise ValueError("Unsafe or duplicate indexed path")
            expected.add(item["path"])
            path = root / name
            path.resolve().relative_to(archive.resolve())
            if path.is_symlink():
                raise ValueError("Archive symlink: " + str(name))
            data = path.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Changed historical file: " + str(name))
            checksums.append(item["sha256"] + "  " + item["path"] + "\n")
            checked += 1
        actual = {str(p.relative_to(root)) for p in archive.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.name != ".DS_Store"}
        if actual != expected or len(expected) != snapshot["file_count"]:
            raise ValueError("Archive file inventory differs")
        manifest = (archive / "PACKAGE-MANIFEST.json").read_bytes()
        if hashlib.sha256(manifest).hexdigest() != snapshot["original_manifest_sha256"]:
            raise ValueError("Historical package manifest changed")
        result = subprocess.run([sys.executable, "-I", "-B", str(archive / "verify_package.py")],
                                check=True, text=True, capture_output=True)
        record = json.loads(result.stdout)
        if record["status"] != "pass":
            raise ValueError("Historical internal manifest failed")
    if (root / "ARCHIVE-SHA256SUMS").read_text() != "".join(checksums):
        raise ValueError("Human-readable checksum list differs")
    print(json.dumps({"status": "passed", "historical_files_checked": checked,
                      "snapshot_count": len(index["snapshots"]),
                      "scope": "Retained snapshot integrity and original package checks only. Not a model run, signature, external replication or current-runtime qualification."}, indent=2))


if __name__ == "__main__":
    main()
