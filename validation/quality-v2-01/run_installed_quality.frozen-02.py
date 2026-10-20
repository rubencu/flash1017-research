#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate frozen quality tasks through an installed oMLX HTTP server.

No inference monkeypatches. The client observes idle runtime configuration and
saves every response. Generated code is only text here; grading is separate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback

MODEL_ID = "DeepSeek-V4.1-Flash-1017"
SEED = 431017


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def request_for(job):
    # Explicit allowlist: test answers, oracles and metadata cannot enter requests.
    messages = job.get("messages")
    if messages is None:
        messages = [{"role": "user", "content": job["prompt"]}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected nonempty text messages")
    if any(set(m) != {"role", "content"} or not isinstance(m["content"], str)
           or m["role"] not in ("system", "user", "assistant") for m in messages):
        raise ValueError("Only explicit text messages are supported")
    return {
        "model": MODEL_ID, "messages": messages,
        "max_tokens": int(job["max_tokens"]), "temperature": 0.0,
        "top_p": 1.0, "top_k": 0, "seed": SEED,
        "repetition_penalty": 1.0, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def configure(args):
    from omlx.settings import GlobalSettings
    from omlx.model_settings import ModelSettings, ModelSettingsManager
    base = args.base_path.resolve()
    if base.exists() and any(base.iterdir()):
        raise ValueError("Each run requires a fresh server state directory")
    if args.arm == "candidate":
        from omlx.flash1017_cli import configure as configure_profile
        configure_profile(args.model, base, args.port)
        settings = GlobalSettings.load(base_path=base)
    else:
        settings = GlobalSettings(base_path=base)
        settings.server.host = "127.0.0.1"
        settings.server.port = args.port
        settings.model.model_dirs = [str(base / "models")]
        settings.model.model_dir = None
        settings.huggingface.hf_cache_enabled = False
        settings.ensure_directories()
        model_link = base / "models" / MODEL_ID
        model_link.parent.mkdir(parents=True, exist_ok=True)
        model_link.symlink_to(args.model.resolve(), target_is_directory=True)
        ModelSettingsManager(base).set_settings(MODEL_ID, ModelSettings(
            deepseek_v41_engram_ssd_offload=True,
            deepseek_v41_ced_prefill_enabled=False,
            mtp_enabled=True, mtp_adaptive_max_depth=3, mtp_fixed_depth=None,
            max_context_window=131072, enable_thinking=False,
            temperature=0.0, top_p=1.0, is_default=True,
        ))
    settings.server.log_level = "INFO"
    settings.scheduler.max_concurrent_requests = 1
    settings.cache.enabled = True
    settings.cache.hot_cache_max_size = "4GB"
    settings.cache.ssd_cache_max_size = "20GB"
    settings.cache.ssd_cache_dir = str(base / "cache")
    settings.sampling.max_context_window = 131072
    settings.sampling.temperature = 0.0
    settings.sampling.top_p = 1.0
    settings.sampling.max_tokens = 2048
    settings.save()
    manager = ModelSettingsManager(base)
    model_settings = manager.get_settings(MODEL_ID)
    model_settings.temperature = 0.0
    model_settings.top_p = 1.0
    model_settings.force_sampling = False
    manager.set_settings(MODEL_ID, model_settings)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    import httpx
    import mlx.core as mx
    import omlx
    import psutil
    from omlx import cli

    expected_version = {"baseline": "0.7.1rc1", "candidate": "0.7.1rc1+flash1017.1"}[args.arm]
    if importlib.metadata.version("omlx") != expected_version:
        raise ValueError("Unexpected installed runtime version")
    if "site-packages/omlx/" not in str(Path(omlx.__file__).resolve()):
        raise ValueError("Use an installed wheel outside the source checkout")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    jobs_path = args.plan.parent / plan["requests_path"]
    jobs_bytes = jobs_path.read_bytes()
    if digest(jobs_bytes) != plan["requests_sha256"]:
        raise ValueError("Frozen requests changed")
    jobs = [json.loads(line) for line in jobs_bytes.splitlines() if line]
    if len(jobs) != plan["tasks"] or len({j['task_id'] for j in jobs}) != len(jobs):
        raise ValueError("Unexpected task inventory")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite or silently resume a result run")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=True)
    if psutil.virtual_memory().available < 210 * 2**30:
        raise MemoryError("At least 210 GiB available memory is needed")
    # Same allocation guidelines for both arms; no altered inference functions.
    device = mx.device_info()
    limit = min(device["max_recommended_working_set_size"], device["memory_size"] - 32 * 2**30)
    mx.set_wired_limit(limit)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(8 * 2**30)
    base = configure(args)
    package = Path(omlx.__file__).parent
    receipt = {
        "schema": "flash1017-quality-generation/1", "arm": args.arm,
        "status": "starting", "plan_sha256": digest(plan_bytes),
        "requests_sha256": digest(jobs_bytes), "total_tasks": len(jobs),
        "completed": 0, "errors": 0, "truncated": 0,
        "runtime_version": expected_version, "runtime_source": "installed wheel",
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "python": sys.version, "device": device, "memory_limit_bytes": limit,
        "native_assets": {str(p.relative_to(package)): digest(p.read_bytes())
            for p in sorted((package / 'custom_kernels').rglob('*'))
            if p.suffix in ('.so', '.metallib')},
        "settings_sha256": {n: digest((base / n).read_bytes())
            for n in ('settings.json', 'model_settings.json')},
        "started_unix": time.time(), "responses": "responses.jsonl",
        "harness_sha256": digest(Path(__file__).read_bytes()),
    }
    receipt_path = args.output / "receipt.json"
    controllers = []

    def save():
        atomic_json(receipt_path, receipt)

    def runtime_snapshot():
        from omlx import server
        entry = server._server_state.engine_pool.get_entry(MODEL_ID)
        engine = entry.engine
        model = engine._vlm_model
        config = engine._engine.engine.scheduler.config
        language = model.language_model
        model_config = language._config
        controller = getattr(model, "_flash1017_runtime_controller", None)
        if controller is not None and controller not in controllers:
            controllers.append(controller)
        confidence = getattr(language, "_flash1017_experiment", None)
        result = {
            "prefill_step_size": config.prefill_step_size,
            "max_num_seqs": config.max_num_seqs,
            "attention_output_quantization": getattr(model_config, "attention_output_quantization", False),
            "flash1017_enabled": getattr(model_config, "flash1017_enabled", False),
            "confidence": None if confidence is None else {"cycles": confidence.cycles,
                "selected_depths": list(confidence.selected_depths)},
            "runtime": None if controller is None else controller.snapshot(),
            "memory_available_gib": psutil.virtual_memory().available / 2**30,
        }
        if args.arm == 'candidate' and (controller is None or confidence is None):
            raise RuntimeError("Candidate profile did not activate")
        if args.arm == 'baseline' and (controller is not None or confidence is not None):
            raise RuntimeError("Baseline has candidate modifications active")
        return result

    def client_work():
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=1800) as client:
                for _ in range(240):
                    try:
                        if client.get('/health').is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Server did not become healthy")
                warmup = client.post('/v1/chat/completions', json=request_for({
                    'prompt': 'Initialize the evaluation. Reply OK.', 'max_tokens': 1}))
                warmup.raise_for_status()
                receipt['warmup'] = {'excluded': True, 'usage': warmup.json().get('usage')}
                receipt['runtime_initial'] = runtime_snapshot()
                receipt['status'] = 'running'
                save()
                with (args.output / 'responses.jsonl').open('x') as stream:
                    for index, job in enumerate(jobs):
                        if digest(jobs_path.read_bytes()) != plan['requests_sha256']:
                            raise RuntimeError('Frozen requests changed during execution')
                        payload = request_for(job)
                        row = {'task_id': job['task_id'], 'suite': job['suite'],
                               'index': index, 'request_sha256': digest(canonical(payload)),
                               'text': '', 'finish_reason': None, 'max_tokens': job['max_tokens']}
                        started = time.monotonic()
                        try:
                            response = client.post('/v1/chat/completions', json=payload)
                            row['http_status'] = response.status_code
                            response.raise_for_status()
                            data = response.json()
                            choice = data['choices'][0]
                            row.update(text=choice['message'].get('content') or '',
                                finish_reason=choice.get('finish_reason'), usage=data.get('usage'),
                                reasoning_content=choice['message'].get('reasoning_content'))
                            if row['finish_reason'] == 'length':
                                receipt['truncated'] += 1
                            if not row['text']:
                                row['error'] = 'Empty assistant content'
                        except Exception as exc:
                            row['error'] = repr(exc)
                        row['wall_seconds'] = time.monotonic() - started
                        row['completed_unix'] = time.time()
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        receipt['completed'] += 1
                        receipt['errors'] += int('error' in row)
                        receipt['last_task_id'] = job['task_id']
                        if index % 20 == 0 or index == len(jobs)-1:
                            receipt['runtime_latest'] = runtime_snapshot()
                        save()
                        print(json.dumps({'arm': args.arm, 'completed': index+1, 'total': len(jobs),
                            'task': job['task_id'], 'finish': row['finish_reason'],
                            'seconds': round(row['wall_seconds'], 3), 'error': row.get('error')}), flush=True)
                        if row.get('error'):
                            raise RuntimeError('Transport/runtime failure retained; stop to investigate')
                        if psutil.virtual_memory().available < 8 * 2**30:
                            raise MemoryError('Available memory below 8 GiB between requests')
                receipt['status'] = 'generated-awaiting-shutdown'
        except BaseException:
            receipt['status'] = 'failed'
            receipt['error'] = traceback.format_exc()
        finally:
            save()
            os.kill(os.getpid(), signal.SIGINT)

    save()
    worker = threading.Thread(target=client_work, name='quality-http-client', daemon=True)
    worker.start()
    try:
        sys.argv = ['omlx', 'serve', '--base-path', str(base), '--host', '127.0.0.1',
                    '--port', str(args.port), '--max-concurrent-requests', '1']
        cli.main()
    except KeyboardInterrupt:
        pass
    finally:
        worker.join(timeout=10)
        receipt['controllers_closed'] = all(c.closed for c in controllers)
        receipt['finished_unix'] = time.time()
        if receipt['status'] == 'generated-awaiting-shutdown' and receipt['controllers_closed']:
            receipt['status'] = 'generated'
        save()
    if receipt['status'] != 'generated':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
