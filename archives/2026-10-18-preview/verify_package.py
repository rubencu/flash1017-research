# SPDX-License-Identifier: Apache-2.0
"""Check this research archive's bytes without importing inference code."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "PACKAGE-MANIFEST.json").read_text())
    if manifest.get("schema") != "flash1017-research-package/1":
        raise ValueError("Unsupported package manifest schema")
    expected, issues = set(), []
    for item in manifest["files"]:
        name = item["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in expected:
            raise ValueError("Unsafe or duplicate manifest path: " + name)
        expected.add(name)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            issues.append("Unsafe file: " + name)
        elif not path.is_file():
            issues.append("Missing file: " + name)
        else:
            data = path.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                issues.append("Changed bytes: " + name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and p.name not in ("PACKAGE-MANIFEST.json", ".DS_Store")}
    extras = sorted(actual - expected)
    if extras:
        issues.extend("Unmanifested file: " + name for name in extras)
    return {"status": "pass" if not issues else "fail", "files_checked": len(expected),
            "issues": issues,
            "scope": "Internal file hashes only; not a signature, external replication or model execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "fail", "issues": [str(exc)]}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
