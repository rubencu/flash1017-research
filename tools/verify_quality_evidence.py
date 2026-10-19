#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a saved quality evidence bundle without executing model or answer code."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(folder):
    folder = folder.resolve()
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    if manifest.get("schema") != "flash1017-quality-evidence/1":
        raise ValueError("Unsupported evidence schema")
    names = set()
    for item in manifest["files"]:
        name = PurePosixPath(item["path"])
        if name.is_absolute() or ".." in name.parts or str(name) in names:
            raise ValueError("Unsafe or duplicate evidence path")
        names.add(str(name))
        path = folder / name
        path.resolve().relative_to(folder)
        if path.is_symlink() or any(
            p.is_symlink() for p in path.parents if p != folder and folder in p.parents
        ):
            raise ValueError("Evidence cannot contain symlinks")
        data = path.read_bytes()
        if (
            len(data) != item["bytes"]
            or hashlib.sha256(data).hexdigest() != item["sha256"]
        ):
            raise ValueError("Changed evidence: " + str(name))
    actual = {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file()
        and p.name not in ("MANIFEST.json", ".DS_Store")
        and "__pycache__" not in p.parts
    }
    if actual != names:
        raise ValueError("Evidence inventory differs")
    return {
        "status": "passed",
        "files_checked": len(names),
        "scope": "Saved evidence identity only; no inference, answer execution, independent replication or universal quality claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder), indent=2))


if __name__ == "__main__":
    main()
