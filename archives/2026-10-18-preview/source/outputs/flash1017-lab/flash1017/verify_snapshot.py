"""Independently hash the pinned downloaded snapshot, including metadata files."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["directory"])
    info = HfApi().model_info(manifest["repo"], revision=manifest["revision"], files_metadata=True)
    if info.sha != manifest["revision"]:
        raise ValueError("Hub returned a different revision")
    report = {"repo": manifest["repo"], "revision": info.sha, "directory": str(root),
              "status": "verifying", "files": {}, "started_at": time.time()}
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old["revision"] == info.sha and old["directory"] == str(root):
            report = old
    while True:
        missing = []
        for file in info.siblings:
            path = root / file.rfilename
            if not path.is_file() or path.stat().st_size != file.size:
                missing.append(file.rfilename)
                continue
            stat = path.stat()
            fingerprint = [stat.st_ino, stat.st_size, stat.st_mtime_ns]
            old = report["files"].get(file.rfilename)
            if old and old["fingerprint"] == fingerprint and old["verified"]:
                continue
            sha256, git = hashlib.sha256(), hashlib.sha1()
            git.update(f"blob {stat.st_size}\0".encode())
            with path.open("rb") as stream:
                while chunk := stream.read(16 * 2**20):
                    sha256.update(chunk)
                    if not file.lfs:
                        git.update(chunk)
            after = path.stat()
            if fingerprint != [after.st_ino, after.st_size, after.st_mtime_ns]:
                raise RuntimeError(f"File changed while hashing: {file.rfilename}")
            expected = file.lfs.sha256 if file.lfs else file.blob_id
            actual = sha256.hexdigest() if file.lfs else git.hexdigest()
            if actual != expected:
                raise ValueError(f"Hash mismatch: {file.rfilename}")
            report["files"][file.rfilename] = {
                "size": stat.st_size, "sha256": sha256.hexdigest(),
                "checked_against": "Hub LFS SHA-256" if file.lfs else "Hub Git blob SHA-1",
                "expected": expected, "verified": True, "fingerprint": fingerprint,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"verified": file.rfilename, "count": len(report["files"])}), flush=True)
        report["missing"] = missing
        if not missing:
            report["status"] = "complete_hash_verified"
            report["completed_at"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not missing or not args.watch:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
