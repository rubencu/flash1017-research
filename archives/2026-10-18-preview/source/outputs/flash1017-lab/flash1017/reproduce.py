"""Rerun a saved experiment with the qualified local code and checkpoint."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['heldout', 'confidence-16k', 'deployment-quality'], default='heldout')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError('Choose a new output name to preserve earlier evidence')
    state = root/'server-state'
    state.mkdir(exist_ok=True)
    with (state/'server.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Stop Flash1017 before running a reproduction; another lab process holds its memory reservation') from exc
        from .setup_runtime import main as setup
        setup()
        profile = json.loads((root/'qualified-profile.json').read_text())
        for name, digest in profile['inference_files_sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Qualified code changed: '+name)
        for name, version in profile['packages'].items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError('Qualified dependency changed: '+name)
        checkpoint = json.loads((root/'results/checkpoint-hashes.json').read_text())
        model = Path(checkpoint['directory'])
        for name, record in checkpoint['files'].items():
            stat = (model/name).stat()
            if [stat.st_ino, stat.st_size, stat.st_mtime_ns] != record['fingerprint']:
                raise RuntimeError('Verified checkpoint changed: '+name)
        sys.argv = [sys.argv[0], '--model', str(model), '--plan', str(root/f'plans/{args.suite}.json'),
                    '--output', str(args.output.resolve())]
        from .bench import main as bench
        bench()


if __name__ == '__main__':
    main()
