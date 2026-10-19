#!/usr/bin/env python3
"""Bounded serving/lifecycle validation of an installed fork, not a benchmark.

Run with the wheel environment's Python and -I from outside the runtime repo.
Uses the real CLI/server and HTTP requests. In-process observation reads idle
controller counters; it does not replace loaders, kernels or generation paths.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    import omlx
    import httpx
    from omlx import cli

    source = str(Path(omlx.__file__).resolve())
    if "site-packages/omlx/" not in source:
        raise RuntimeError(
            "Run against the installed wheel, outside the source checkout"
        )
    receipt = {
        "kind": "installed-wheel-serving-smoke",
        "performance_benchmark": False,
        "quality_evaluation": False,
        "runtime_import": "site-packages/omlx/"
        + source.split("site-packages/omlx/", 1)[1],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("omlx", "mlx", "mlx-lm", "mlx-vlm")
        },
        "requests": [],
        "status": "running",
    }
    controllers = []
    for package in (
        "bonsai",
        "decode_fast",
        "glm_moe_dsa",
        "minimax_m3",
        "qwen35_prefill",
    ):
        importlib.import_module(f"omlx.custom_kernels.{package}._ext")
    receipt["native_modules_imported"] = 5
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")

    def exercise():
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{args.port}", timeout=900
            ) as client:
                for _ in range(240):
                    try:
                        if client.get("/health").is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Server did not become healthy")
                from omlx import server

                model_id = "DeepSeek-V4.1-Flash-1017"

                def payload(prompt, **kwargs):
                    return {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 32,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        **kwargs,
                    }

                def chat(name, prompt, **kwargs):
                    response = client.post(
                        "/v1/chat/completions", json=payload(prompt, **kwargs)
                    )
                    response.raise_for_status()
                    data = response.json()
                    text = data["choices"][0]["message"].get("content", "")
                    if not text:
                        raise RuntimeError(f"Empty response for {name}: {data}")
                    receipt["requests"].append(
                        {
                            "name": name,
                            "http_status": response.status_code,
                            "usage": data.get("usage"),
                            "text": text,
                            "finish_reason": data["choices"][0].get("finish_reason"),
                        }
                    )
                    save()

                chat("short", "Answer in one sentence: What is two plus two?")
                entry = server._server_state.engine_pool.get_entry(model_id)
                engine = entry.engine
                model = engine._vlm_model
                controller = model._flash1017_runtime_controller
                controllers.append(controller)
                receipt["memory_setup"] = model._flash1017_memory_receipt
                config = engine._engine.engine.scheduler.config
                receipt["scheduler"] = {
                    key: getattr(config, key)
                    for key in (
                        "max_num_seqs",
                        "completion_batch_size",
                        "prefill_step_size",
                        "max_total_tokens",
                        "paged_ssd_cache_dir",
                    )
                }
                receipt["scheduler"]["paged_ssd_cache_dir"] = str(
                    Path(config.paged_ssd_cache_dir).relative_to(
                        Path(args.base_path).expanduser().resolve()
                    )
                )
                tokenizer = engine._tokenizer
                corpus = Path(args.corpus).read_text()
                excerpt = tokenizer.decode(tokenizer.encode(corpus)[:8300])
                prompt = excerpt + "\nSummarize the code above in two short sentences."
                chat("wide-prefill", prompt)
                chat("prefix-reuse", prompt + " Focus on error handling.")
                receipt["runtime"] = controller.snapshot()
                confidence = model.language_model._flash1017_experiment
                receipt["confidence"] = {
                    "cycles": confidence.cycles,
                    "selected_depths": list(confidence.selected_depths),
                }
                receipt["prefetch_budget_bytes"] = (
                    model.language_model._engram_prefetch.budget_bytes
                )
                if not receipt["runtime"]["enabled"] or not confidence.cycles:
                    raise RuntimeError("Optimized profile did not activate")
                if not receipt["runtime"]["cone"]["cone_counts"]:
                    raise RuntimeError("Wide-prefill replay did not activate")
                save()
                oversized = " test" * 132000
                if len(tokenizer.encode(oversized)) <= 131072:
                    raise RuntimeError(
                        "Context-limit fixture is unexpectedly too short"
                    )
                # Drop model/engine references before ordinary server shutdown.
                del confidence, model, engine, tokenizer, entry
                response = client.post(
                    "/v1/chat/completions", json=payload(oversized, max_tokens=1)
                )
                receipt["context_limit_http_status"] = response.status_code
                if response.status_code != 400:
                    raise RuntimeError("Total context cap was not enforced")
                response = client.post(
                    "/v1/chat/completions", json=payload("Hello", max_tokens=131073)
                )
                receipt["total_budget_http_status"] = response.status_code
                if (
                    response.status_code != 400
                    or response.json().get("error", {}).get("code")
                    != "context_length_exceeded"
                ):
                    raise RuntimeError("Prompt plus output budget cap was not enforced")
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload(
                        "Count upwards from one, one number per line.",
                        max_tokens=256,
                        stream=True,
                    ),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            item = json.loads(line[6:])
                            if (
                                item.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            ):
                                receipt["stream_cancelled_after_content"] = True
                                break
                chat(
                    "after-cancellation",
                    "Answer briefly: What is the capital of France?",
                )
                receipt["status"] = "requests-passed-awaiting-shutdown"
        except BaseException:
            receipt["status"] = "failed"
            receipt["error"] = traceback.format_exc()
        finally:
            save()
            os.kill(os.getpid(), signal.SIGINT)

    worker = threading.Thread(
        target=exercise, name="flash1017-smoke-client", daemon=True
    )
    worker.start()
    try:
        cli.main(
            [
                "flash1017",
                "--model",
                args.model,
                "--base-path",
                args.base_path,
                "--port",
                str(args.port),
            ]
        )
    except KeyboardInterrupt:
        pass
    finally:
        worker.join(timeout=10)
        receipt["controllers_closed"] = bool(controllers) and all(
            c.closed for c in controllers
        )
        if (
            receipt["status"] == "requests-passed-awaiting-shutdown"
            and receipt["controllers_closed"]
        ):
            receipt["status"] = "passed"
        save()
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
