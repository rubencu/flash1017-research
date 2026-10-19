"""Verify the archival source kit without importing MLX or contacting a server."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default = here.parent if here.name == "source" else here.parent / "package"
    parser.add_argument("--package", type=Path, default=default)
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    inventory = json.loads((package / "SOURCE-INVENTORY.json").read_text())
    originals_checked = 0
    for record in inventory["files"]:
        public = within(package, record["path"])
        if sha(public) != record["public_sha256"] or public.stat().st_size != record["public_bytes"]:
            raise ValueError("Source copy differs: " + record["path"])
        if public.suffix == ".py":
            ast.parse(public.read_text(), filename=record["path"])
        if args.original_root and record["original"].startswith("${ORIGINAL_LAB_ROOT}/"):
            original = within(args.original_root, record["original"].split("/", 1)[1])
            if sha(original) != record["original_sha256"]:
                raise ValueError("Original changed: " + record["original"])
            originals_checked += 1
    for record in inventory["generated_files"]:
        path = within(package, record["path"])
        if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Generated source artifact differs: " + record["path"])
    expected = {r["path"] for r in inventory["files"]} | {r["path"] for r in inventory["generated_files"]}
    actual = {str(p.relative_to(package)) for p in (package / "source").rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError({"missing": sorted(expected - actual), "unlisted": sorted(actual - expected)})
    receipt = inventory["runtime_patch"]
    with tempfile.TemporaryDirectory(prefix="flash1017-source-check-") as directory:
        temp = Path(directory)
        shutil.copytree(package / "source/upstream-baseline/omlx", temp / "omlx")
        subprocess.run(["patch", "-p1", "-F", "0", "--batch", "-i", str(within(package, receipt["patch_path"]))],
                       cwd=temp, check=True, capture_output=True, text=True)
        for record in receipt["files"]:
            if sha(within(temp / "omlx", record["path"])) != record["candidate_sha256"]:
                raise ValueError("Patched source differs: " + record["path"])
    audit = json.loads((package / "source/SOURCE-CLOSURE-AUDIT.json").read_text())
    if audit["missing_reachable_local_sources"] or audit["archived_local_source_gaps"]:
        raise ValueError("The recorded local source audit has unresolved code edges")
    profile = json.loads((package / "source/outputs/flash1017-lab/qualified-candidate-profile.json").read_text())
    policy = json.loads((package / "source/outputs/flash1017-lab/results/confidence-policy6-01.json").read_text())
    if profile["runtime"]["policy"] != {key: policy[key] for key in ("cycle_ms", "temperatures")}:
        raise ValueError("Frozen calibration does not match the candidate profile")
    print(json.dumps({"status": "passed", "copied_files": len(inventory["files"]),
        "generated_files": len(inventory["generated_files"]), "originals_checked": originals_checked,
        "five_file_patch_reconstruction": "byte-identical", "local_code_edges_recorded": "no missing targets",
        "scope": "Integrity, Python syntax, recorded source-edge inventory, frozen policy values and offline patch reconstruction only. No external build, inference, serving or performance qualification."}, indent=2))


if __name__ == "__main__":
    main()
