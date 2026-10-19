"""Serve the qualified local profile through an isolated oMLX instance."""
import argparse
import hashlib
import fcntl
import socket
import json
import os
from pathlib import Path
import sys


def check_port_available(port):
    with socket.socket() as check:
        # Match the HTTP server's address-reuse behavior. A closed connection
        # in TIME_WAIT must not look like a live listener after Stop.
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            check.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use; no model was loaded") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile_path = args.profile or root / "qualified-profile.json"
    if not profile_path.is_file():
        raise RuntimeError("A qualified full-model profile has not been selected yet")
    profile = json.loads(profile_path.read_text())
    if not profile.get("qualified", False):
        raise RuntimeError("This profile has not passed full-model qualification")
    from .setup_runtime import main as setup
    setup()
    # Qualification is tied to the code that was measured. Documentation can
    # change freely; inference code requires another explicit validation.
    for relative, expected in profile.get("inference_files_sha256", {}).items():
        source = root / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Qualified inference code changed: {relative}; revalidate it first")
    hashes = json.loads((root / "results/checkpoint-hashes.json").read_text())
    if hashes["status"] != "complete_hash_verified":
        raise RuntimeError("The pinned checkpoint has not passed file verification")
    model_path = Path(hashes["directory"]).resolve()
    for name, record in hashes["files"].items():
        stat = (model_path / name).stat()
        if [stat.st_ino, stat.st_size, stat.st_mtime_ns] != record["fingerprint"]:
            raise RuntimeError(f"Verified checkpoint changed: {name}; verify it again")
    import importlib.metadata
    provenance = json.loads((root / "provenance.json").read_text())
    packages = profile.get("packages", {"mlx": provenance["mlx_version"]})
    for package, expected_version in packages.items():
        if importlib.metadata.version(package) != expected_version:
            raise RuntimeError(f"{package} changed; revalidate the runtime first")

    # One heavy model process per lab, including different profile namespaces.
    state_root = root / "server-state"
    state_root.mkdir(exist_ok=True)
    process_lock = (state_root / "server.lock").open("a+")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Flash1017 is already running. Open http://127.0.0.1:8001/admin/chat")
        return
    check_port_available(args.port)
    import psutil
    if psutil.virtual_memory().available < 210 * 2**30:
        raise MemoryError("Flash1017 needs room for about 205 GiB of weights. Unload other models before starting it.")

    runtime = profile["runtime"]
    identity = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()[:12]
    # The opt-in reference rounding changes hidden states without editing the
    # checkpoint config. Never reuse prefix caches from a different profile.
    state_root = root / "server-state"
    base = state_root / identity
    models = base / "models"
    models.mkdir(parents=True, exist_ok=True)
    link = models / model_path.name
    if not link.exists():
        link.symlink_to(model_path, target_is_directory=True)
    elif link.resolve() != model_path:
        raise RuntimeError("Server model link points to a different checkpoint")
    os.environ["OMLX_BASE_PATH"] = str(base)
    from omlx.settings import GlobalSettings
    from omlx.model_settings import ModelSettings, ModelSettingsManager
    if not (base / "settings.json").exists():
        settings = GlobalSettings(base_path=base)
        settings.server.host, settings.server.port = "127.0.0.1", args.port
        settings.model.model_dirs = [str(models)]
        settings.scheduler.max_concurrent_requests = 1
        settings.auth.skip_api_key_verification = True  # This server binds only to loopback.
        # The bundled chat page uses a nonempty key to initialize its model
        # picker even when localhost authentication is disabled.
        import secrets
        settings.auth.api_key = secrets.token_urlsafe(32)
        settings.cache.hot_cache_max_size = "4GB"
        settings.cache.ssd_cache_max_size = "20GB"
        settings.save()
    original_scheduler_config = GlobalSettings.to_scheduler_config

    def scheduler_config(settings):
        config = original_scheduler_config(settings)
        config.prefill_step_size = int(runtime.get("prefill_step", 2048))
        return config

    GlobalSettings.to_scheduler_config = scheduler_config
    manager = ModelSettingsManager(base)
    manager.set_settings(model_path.name, ModelSettings(
        is_default=True, is_pinned=True,
        display_name="DeepSeek V4.1 Flash 1017 · Local Lab",
        description="Verified October 17 checkpoint with a locally tested vocabulary projection kernel.",
        mtp_enabled=runtime.get("mtp", True),
        mtp_adaptive_max_depth=runtime.get("depth", 3),
        mtp_fixed_depth=runtime.get("depth", 3) if runtime.get("fixed", False) else None,
        deepseek_v41_engram_ssd_offload=True,
        deepseek_v41_ced_prefill_enabled=runtime.get("ced_prefill", False),
        max_context_window=profile["max_context_window"],
        temperature=1.0, top_p=1.0, enable_thinking=False,
    ))

    from omlx.patches.deepseek_v41 import loading
    from .bench import configure
    original_load = loading.load

    def load_target(path, **kwargs):
        if Path(path).resolve() != model_path:
            raise ValueError("This experiment server only loads the verified 1017 checkpoint")
        kwargs["engram_ssd_offload"] = True
        kwargs["attention_output_quantization"] = runtime.get("attention_rounding", False)
        from .memory import prepare_memory
        prepare_memory(base / "load-memory.jsonl")
        model, processor = original_load(path, **kwargs)
        experiment = configure(model, runtime)
        if experiment is not None:
            experiment.record_events = False
        return model, processor

    loading.load = load_target
    import psutil
    stamp = state_root / "server-process.json"
    stamp.write_text(json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time(),
                                 "port": args.port, "profile": str(profile_path.resolve())}) + "\n")
    sys.argv = ["omlx", "serve", "--base-path", str(base), "--model-dir", str(models),
                "--host", "127.0.0.1", "--port", str(args.port)]
    from omlx.cli import main as cli_main
    try:
        cli_main()
    finally:
        stamp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
