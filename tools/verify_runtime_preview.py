# SPDX-License-Identifier: Apache-2.0
"""Check preview evidence and optionally the wheel; no imports of runtime code."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    link = json.loads((root / "RUNTIME-LINK.json").read_text())
    folder = (root / link["validation"]).resolve()
    folder.relative_to(root)
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    if manifest["runtime_commit"] != link["runtime_commit"]:
        raise ValueError("Runtime source identity differs")
    if manifest["wheel_sha256"] != link["runtime_wheel_sha256"]:
        raise ValueError("Wheel identity differs")
    names = set()
    for entry in manifest["files"]:
        name = entry["path"]
        if name in names:
            raise ValueError("Duplicate evidence path")
        names.add(name)
        path = folder / name
        path.resolve().relative_to(folder)
        if path.is_symlink():
            raise ValueError("Evidence must not be a symlink")
        data = path.read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Evidence differs: " + name)
    actual = {
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.json", ".DS_Store"}
    }
    if actual != names:
        raise ValueError("Evidence inventory differs")
    if (
        args.wheel
        and hashlib.sha256(args.wheel.read_bytes()).hexdigest()
        != link["runtime_wheel_sha256"]
    ):
        raise ValueError("Supplied wheel differs")
    print(
        json.dumps(
            {
                "status": "passed",
                "files_checked": len(names),
                "runtime_commit": link["runtime_commit"],
                "wheel_checked": bool(args.wheel),
                "scope": "Evidence integrity and optional artifact identity; not inference, a benchmark or an independent signature.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
