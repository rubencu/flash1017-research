#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check an installed Flash1017.2 wheel and measured dependency versions.

Run with the installation's Python. Does not load a model or import runtime code.
Dependency artifact hashes and build-tool environments are outside this check.
"""

import argparse
import email.parser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--constraints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = root / "validation/performance-v2-01/qualified/block-0-candidate/receipt.json"
    identity = json.loads((root / "validation/quality-v2-01/BUILD-IDENTITY.json").read_text())
    expected = json.loads(receipt.read_text())["shared_environment"]["packages"]
    errors = []
    constraints = {}
    for line in args.constraints.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        name = canonical(name)
        if name in constraints:
            errors.append("Duplicate constraint: " + name)
        constraints[name] = version
    if constraints != expected:
        errors.append("Constraints differ from measured package versions")
    installed = {canonical(d.metadata["Name"]): d.version
                 for d in importlib.metadata.distributions()}
    for name, version in expected.items():
        if installed.get(name) != version:
            errors.append(f"Dependency mismatch: {name}: {installed.get(name)!r} != {version!r}")
    candidate = identity["candidate"]
    wheel_sha = sha(args.wheel.read_bytes())
    if wheel_sha != candidate["wheel_sha256"]:
        errors.append("Wheel checksum differs from measured wheel")
    if installed.get("omlx") != candidate["version"]:
        errors.append("Installed oMLX version differs")
    if sys.version_info[:2] != (3, 12):
        errors.append("Requires CPython 3.12")
    dist = importlib.metadata.distribution("omlx")
    checked = 0
    git_dependencies = {}
    with zipfile.ZipFile(args.wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith(("omlx/", "omlx_web/")) and not name.endswith("/"):
                target = Path(dist.locate_file(name))
                if not target.is_file() or sha(target.read_bytes()) != sha(wheel.read(name)):
                    errors.append("Installed file differs: " + name)
                checked += 1
        metadata_name, = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = email.parser.BytesParser().parsebytes(wheel.read(metadata_name))
        for requirement in metadata.get_all("Requires-Dist", []):
            match = re.fullmatch(r"([\w.-]+)\s*@\s*git\+(https://[^\s@]+)@([0-9a-f]{40})", requirement)
            if not match:
                continue
            name, url, commit = match.groups()
            dep = importlib.metadata.distribution(name)
            direct = json.loads(dep.read_text("direct_url.json") or "{}")
            vcs = direct.get("vcs_info", {})
            if direct.get("url") != url or vcs.get("commit_id") != commit or vcs.get("vcs") != "git":
                errors.append("Git source identity differs: " + name)
            git_dependencies[canonical(name)] = {"url": url, "commit": commit}
    if checked != 752 or len(git_dependencies) != 4:
        errors.append("Unexpected measured package or Git dependency inventory")
    pip_check = subprocess.run([sys.executable, "-I", "-m", "pip", "check"],
                               text=True, capture_output=True)
    if pip_check.returncode:
        errors.append("pip check failed")
    record = {
        "schema": "flash1017-release-install/1",
        "status": "passed" if not errors else "failed",
        "scope": "Fresh virtual environment installation; no model generation, new quality evaluation or performance measurement.",
        "python": sys.version,
        "python_matches_measured": sys.version == json.loads(receipt.read_text())["shared_environment"]["python"],
        "runtime_version": installed.get("omlx"),
        "wheel_sha256": wheel_sha,
        "constraints_sha256": sha(args.constraints.read_bytes()),
        "source_receipt_sha256": sha(receipt.read_bytes()),
        "installed_package_files_checked": checked,
        "dependency_versions_checked": len(expected),
        "installed_dependency_versions": {n: installed.get(n) for n in expected},
        "additional_packages": {n: v for n, v in installed.items() if n not in expected and n != "omlx"},
        "git_dependencies": git_dependencies,
        "pip_check": {"exit_code": pip_check.returncode, "stdout": pip_check.stdout, "stderr": pip_check.stderr},
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "errors": errors,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "installed_package_files_checked", "dependency_versions_checked", "python_matches_measured", "additional_packages", "errors")}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
