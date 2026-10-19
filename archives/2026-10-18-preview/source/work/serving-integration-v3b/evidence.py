"""CPU-only provenance checks for the persistent-serving qualification round.

This round leaves the qualified baseline unchanged. New implementations and
plans are frozen separately before any full-model run, and every runner holds
the same process lock as the existing local server.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "outputs/flash1017-lab"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_baseline():
    profile_path = LAB / "qualified-profile.json"
    profile = json.loads(profile_path.read_text())
    if profile.get("qualified") is not True:
        raise ValueError("Expected the unchanged qualified native64 baseline")
    runtime = profile["runtime"]
    if (runtime.get("ced_prefill") is not False
            or runtime.get("prefill_step") != 8192
            or runtime.get("engram_prefetch_bytes") != 64 * 2**20):
        raise ValueError("Expected native64 with full 8192-row prefill")
    for relative, digest in profile["inference_files_sha256"].items():
        if sha(LAB / relative) != digest:
            raise ValueError("Qualified source changed: " + relative)
    for package, version in profile["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Qualified dependency changed: " + package)
    checkpoint_path = LAB / "results/checkpoint-hashes.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint.get("status") != "complete_hash_verified" or checkpoint.get("missing"):
        raise ValueError("Checkpoint verification is incomplete")
    model_path = Path(checkpoint["directory"])
    for name, record in checkpoint["files"].items():
        if record.get("verified") is not True:
            raise ValueError("Unverified checkpoint file: " + name)
        stat = (model_path / name).stat()
        if [stat.st_ino, stat.st_size, stat.st_mtime_ns] != record["fingerprint"]:
            raise ValueError("Verified checkpoint file changed: " + name)
    return profile, model_path, {
        "profile_sha256": sha(profile_path),
        "inference_files_verified": len(profile["inference_files_sha256"]),
        "checkpoint_manifest_sha256": sha(checkpoint_path),
        "checkpoint_fingerprints_verified": len(checkpoint["files"]),
        "packages": profile["packages"],
        "checkpoint_check": "Inode, size and mtime compared to previously fully hashed files.",
    }


def validate_frozen(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    for path, digest in manifest["sources_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Frozen source changed: " + path)
    profile, model_path, baseline = verify_baseline()
    if baseline != manifest["baseline"]:
        raise ValueError("Qualified baseline changed since the plan was frozen")
    return manifest, profile, model_path


def freeze(manifest_path, paths, *, description):
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError("Use a new manifest; do not replace frozen evidence")
    _, _, baseline = verify_baseline()
    paths = sorted({str(Path(path).resolve()) for path in paths})
    record = {"description": description, "baseline": baseline,
              "sources_sha256": {path: sha(path) for path in paths}}
    write_json(manifest_path, record)
    return record


def snapshot(manifest_path, directory):
    manifest, _, _ = validate_frozen(manifest_path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = manifest["sources_sha256"] | {str(Path(manifest_path).resolve()): sha(manifest_path)}
    records = {}
    for index, (source, digest) in enumerate(sorted(paths.items())):
        source = Path(source)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Source changed during snapshot: " + str(source))
        name = f"{index:03d}-{source.name}"
        (directory / name).write_bytes(data)
        records[str(source)] = {"sha256": digest, "snapshot": name}
    write_json(directory / "manifest.json", records)
    return records


@contextmanager
def exclusive_model():
    with (LAB / "server-state/server.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_fresh_output(output, suffixes):
    paths = [Path(output)] + [Path(output).with_suffix(s) for s in suffixes]
    if any(path.exists() for path in paths):
        raise FileExistsError("Use a fresh output stem to preserve previous evidence")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
