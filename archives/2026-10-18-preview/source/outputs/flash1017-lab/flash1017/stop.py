"""Stop only the process this lab started, protecting against reused PIDs."""
import json
from pathlib import Path
import urllib.parse
import urllib.request
import psutil


def main():
    stamp = Path(__file__).resolve().parents[1] / "server-state/server-process.json"
    if not stamp.exists():
        print("Flash1017 server is not running.")
        return
    record = json.loads(stamp.read_text())
    try:
        process = psutil.Process(record["pid"])
        if process.create_time() != record["created"]:
            raise RuntimeError("Saved PID belongs to another process; it was not stopped")
        # The server's unload path first aborts active inference leases. This
        # lets Stop interrupt a long prefill instead of waiting for its reply.
        root = stamp.parents[1]
        checkpoint = json.loads((root / "results/checkpoint-hashes.json").read_text())
        model_id = urllib.parse.quote(Path(checkpoint["directory"]).name, safe="")
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{record['port']}/v1/models/{model_id}/unload",
                data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=5):
                pass
        except OSError:
            pass  # A starting or already exiting server may not answer yet.
        process.terminate()
        process.wait(timeout=60)
    except psutil.NoSuchProcess:
        pass
    stamp.unlink(missing_ok=True)
    print("Flash1017 server stopped.")


if __name__ == "__main__":
    main()
