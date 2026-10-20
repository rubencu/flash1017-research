"""Prepare CPU-only, reviewable HTTP performance fixtures; never run a server.

This produces a draft, not permission to start the benchmark. Historical source
token streams are decoded and independently rendered through the pinned native
encoder. Any retokenization difference is recorded, never hidden by token splice.
"""

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = (
    ROOT
    / "repositories/flash1017-research/archives/2026-10-18-preview/evidence/overall"
)
MODEL = Path(
    "/Users/rubencu/Library/Application Support/LocalAI/models/"
    "DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp"
)
ENCODER = ROOT / "repositories/omlx/omlx/patches/deepseek_v41/encoding.py"
QUALITY_HARNESS = (
    ROOT / "repositories/flash1017-research/tools/run_installed_quality.py"
)
ORDER = [
    "held-code-1k",
    "performance-cpython-16384",
    "performance-pride-prejudice-65536",
    "held-prose-1k",
    "performance-curl-16384",
    "performance-cpython-65536",
    "performance-moby-dick-16384",
    "performance-pride-prejudice-16384",
]
WARMUP = "authored-warmup-16384"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def token_sha(tokens):
    # Match archived token-stream hash serialization, including JSON spaces.
    return sha(json.dumps(tokens).encode())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def assert_cpu_only():
    forbidden = [
        name
        for name in sys.modules
        if name in {"mlx", "omlx", "torch"}
        or name.startswith(("mlx.", "omlx.", "torch."))
    ]
    if forbidden:
        raise RuntimeError(f"Inference modules imported: {forbidden}")


def request_factory():
    # Execute only the existing payload allowlist and its literal constants.
    # Importing the whole generation harness is unnecessary for fixture prep.
    source = QUALITY_HARNESS.read_text()
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "request_for"
    )
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"MODEL_ID", "SEED"}:
                    constants[target.id] = ast.literal_eval(node.value)
    if set(constants) != {"MODEL_ID", "SEED"}:
        raise ValueError("Missing quality request constants")
    scope = dict(constants)
    exec(  # noqa: S102 -- execute only our checked-in payload allowlist AST
        compile(ast.Module(body=[fn], type_ignores=[]), str(QUALITY_HARNESS), "exec"),
        scope,
    )
    return scope["request_for"], {
        "source": str(QUALITY_HARNESS.relative_to(ROOT)),
        "source_sha256_at_preparation": sha(source.encode()),
        "request_for_ast_sha256": sha(ast.dump(fn, include_attributes=False).encode()),
        "constants": constants,
        "performance_overrides": {
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    }


def main():
    from tokenizers import Tokenizer

    assert_cpu_only()
    if (HERE / "plan-draft-01.json").exists():
        raise FileExistsError(
            "Refusing to replace an existing draft. Review before revising."
        )
    archive_plan_path = ARCHIVE / "plan-01.json"
    archived = json.loads(archive_plan_path.read_text())
    manifest_path = ARCHIVE / "fixtures/fixture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert sha(ENCODER.read_bytes()) == manifest["native_encoder_sha256"]
    tokenizer_path = MODEL / "tokenizer.json"
    assert (
        sha(tokenizer_path.read_bytes())
        == "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    spec = importlib.util.spec_from_file_location(
        "performance_fixture_native_encoding", ENCODER
    )
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    sentinel = "PERFORMANCE_FIXTURE_BOUNDARY_431017"
    empty_frame = encoder.encode_messages(
        [{"role": "user", "content": sentinel}], thinking_mode="chat"
    )
    before, after = empty_frame.split(sentinel)
    assert before and after
    request_for, request_identity = request_factory()
    special_ids = {
        row["id"]
        for row in json.loads(tokenizer_path.read_text())["added_tokens"]
        if row.get("special")
    }
    jobs, payloads, rendered, provenance = [], [], [], []
    for old_id in [WARMUP, *ORDER]:
        old = archived["prompts"][old_id]
        assert token_sha(old["tokens"]) == old["input_tokens_sha256"]
        text = tokenizer.decode(old["tokens"], skip_special_tokens=False)
        assert text.startswith(before) and text.endswith(after)
        content = text[len(before) : -len(after)]
        assert (
            "\ufffd" not in content
        ), "Token decoding may have truncated a Unicode sequence"
        assert content.endswith(old["text"])
        content_ids = tokenizer.encode(content, add_special_tokens=False).ids
        assert not special_ids.intersection(content_ids)
        messages = [{"role": "user", "content": content}]
        rendered_text = encoder.encode_messages(messages, thinking_mode="chat")
        assert rendered_text == text
        new_ids = tokenizer.encode(rendered_text, add_special_tokens=False).ids
        source_record = None
        if old.get("source"):
            assert content.endswith("\n\n" + old["text"])
            source_record = manifest["corpora"][old["source"]]
            corpus_path = ARCHIVE / "fixtures" / source_record["path"]
            assert sha(corpus_path.read_bytes()) == source_record["sha256"]
            corpus_part = content[: -len("\n\n" + old["text"])]
            assert corpus_path.read_text().startswith(corpus_part)
        exact = new_ids == old["tokens"]
        differences = [
            i for i, (a, b) in enumerate(zip(new_ids, old["tokens"])) if a != b
        ]
        warmup = old_id == WARMUP
        task_id = "http-" + old_id
        job = {
            "task_id": task_id,
            "suite": "excluded-warmup" if warmup else "installed-http-performance",
            "messages": messages,
            "max_tokens": 64 if warmup else 512,
            "warmup": warmup,
            "expected_prompt_tokens": len(new_ids),
            "expected_prompt_tokens_sha256": token_sha(new_ids),
        }
        payload = request_for(job)
        payload.update(stream=True, stream_options={"include_usage": True})
        jobs.append(job)
        payloads.append(
            {
                "task_id": task_id,
                "warmup": warmup,
                "request": payload,
                "request_sha256": sha(canonical(payload)),
            }
        )
        rendered.append(
            {"task_id": task_id, "tokens": new_ids, "tokens_sha256": token_sha(new_ids)}
        )
        provenance.append(
            {
                "task_id": task_id,
                "historical_id": old_id,
                "cohort": old["cohort"],
                "source": old.get("source", "authored-synthetic"),
                "source_corpus": (
                    None
                    if source_record is None
                    else {
                        "path": str(
                            (ARCHIVE / "fixtures" / source_record["path"]).relative_to(
                                ROOT
                            )
                        ),
                        "sha256": source_record["sha256"],
                    }
                ),
                "synthetic": old.get("source") is None,
                "max_tokens": job["max_tokens"],
                "historical_prompt_tokens": len(old["tokens"]),
                "historical_tokens_sha256": token_sha(old["tokens"]),
                "actual_http_prompt_tokens": len(new_ids),
                "actual_http_tokens_sha256": token_sha(new_ids),
                "messages_sha256": sha(canonical(messages)),
                "user_text_sha256": sha(content.encode()),
                "rendered_text_sha256": sha(rendered_text.encode()),
                "historical_text_reconstructed_exactly": True,
                "historical_token_stream_reconstructed_exactly": exact,
                "first_token_difference": (
                    differences[0]
                    if differences
                    else (None if exact else min(len(new_ids), len(old["tokens"])))
                ),
                "construction": (
                    "Exact historical tokens recovered as a single user message and round-tripped through the pinned native chat encoder."
                    if exact
                    else "New HTTP adaptation: identical decoded historical corpus excerpt and question, rerendered as normal user text. Ordinary BPE retokenization changes the historical spliced token stream; new HTTP tokens are authoritative."
                ),
            }
        )
    write_jsonl(HERE / "requests-draft-01.jsonl", jobs)
    write_jsonl(HERE / "payloads-draft-01.jsonl", payloads)
    write_jsonl(HERE / "rendered-inputs-draft-01.jsonl", rendered)
    provenance_file = HERE / "fixture-provenance-draft-01.json"
    write_json(
        provenance_file,
        {
            "schema": 1,
            "status": "draft_pending_parent_review_no_model_execution",
            "historical_plan": {
                "path": str(archive_plan_path.relative_to(ROOT)),
                "sha256": sha(archive_plan_path.read_bytes()),
            },
            "historical_fixture_manifest": {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": sha(manifest_path.read_bytes()),
            },
            "source_manifest": {
                "path": str(
                    (ARCHIVE / "fixtures/source-manifest.json").relative_to(ROOT)
                ),
                "sha256": sha((ARCHIVE / "fixtures/source-manifest.json").read_bytes()),
            },
            "attribution_notice": str((ARCHIVE / "fixtures/NOTICE").relative_to(ROOT)),
            "redistribution_note": "Archive sources, original licenses and NOTICE must accompany redistributed excerpts; do not ship this work directory alone as a corpus bundle.",
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": sha(tokenizer_path.read_bytes()),
                "tokenizers_version": importlib.metadata.version("tokenizers"),
            },
            "encoder": {
                "path": str(ENCODER.relative_to(ROOT)),
                "sha256": sha(ENCODER.read_bytes()),
            },
            "request_factory": request_identity,
            "fixtures": provenance,
            "cpu_only": True,
            "inference_modules_loaded": False,
            "weights_opened": False,
        },
    )
    quality_plan = json.loads(
        (ROOT / "work/flash1017-quality-01/protocol-02.json").read_text()
    )
    baseline_receipt = json.loads(
        (ROOT / "work/flash1017-quality-01/baseline-02/receipt.json").read_text()
    )
    processes = []
    for block, arms in enumerate(
        (("baseline", "candidate"), ("candidate", "baseline"))
    ):
        case_order = ORDER if block == 0 else list(reversed(ORDER))
        for arm in arms:
            processes.append(
                {
                    "process_index": len(processes),
                    "process_id": f"block-{block}-{arm}",
                    "block": block,
                    "arm": arm,
                    "fresh_process": True,
                    "fresh_state_directory": True,
                    "warmup_task_id": "http-" + WARMUP,
                    "measured_task_ids": ["http-" + name for name in case_order],
                }
            )
    artifacts = {}
    for name in (
        "requests-draft-01.jsonl",
        "payloads-draft-01.jsonl",
        "rendered-inputs-draft-01.jsonl",
        "fixture-provenance-draft-01.json",
    ):
        path = HERE / name
        artifacts[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path.read_bytes()),
        }
    plan = {
        "schema": "installed-http-performance-v1-draft",
        "status": "draft_pending_parent_review_no_model_execution",
        "scope": "Fresh installed public baseline wheel versus the final stock-arithmetic Flash1017 candidate wheel through public HTTP chat completions, on one machine, with the model loaded and prefix reuse disabled.",
        "prerequisites": [
            "Final candidate must pass the separately frozen 618-task quality/parity gate before any performance run.",
            "Parent reviews this draft, binds candidate wheel/source and quality-result hashes, and freezes harness, plan, fixtures and settings before first output.",
            "Baseline/candidate use the same checkpoint, dependency versions, native assets, Python, hardware and memory limits. Only oMLX wheel and declared profile settings differ.",
            "No simultaneous model generation, model grading or other intentional compute-heavy task during timed processes.",
        ],
        "model": quality_plan["model"],
        "baseline": quality_plan["baseline"],
        "candidate": {
            "status": "pending_final_safe_v2_identity_and_quality_gate",
            "wheel_sha256": None,
            "source_commit": None,
            "installed_version": None,
        },
        "quality_receipt_sha256": None,
        "performance_harness_sha256": None,
        "observed_reference_device": baseline_receipt["device"],
        "memory_limit_bytes": baseline_receipt["memory_limit_bytes"],
        "artifacts": artifacts,
        "request_factory": request_identity,
        "http": {
            "endpoint": "/v1/chat/completions",
            "host": "127.0.0.1",
            "stream": True,
            "stream_options": {"include_usage": True},
            "concurrency": 1,
            "client_clock": "time.perf_counter_ns",
            "request_timeout_seconds": 1200,
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 431017,
            "enable_thinking": False,
            "max_tokens_measured": 512,
            "max_tokens_warmup": 64,
            "ordinary_eos": True,
            "force_length": False,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "processes": processes,
        "counts": {
            "processes": 4,
            "blocks": 2,
            "measured_requests": 32,
            "paired_case_block_observations": 16,
            "excluded_warmups": 4,
            "unique_measured_workloads": 8,
            "synthetic_1k": 2,
            "natural_16k": 4,
            "natural_64k": 2,
        },
        "cache_policy": {
            "prefix_reuse": "disabled through normal settings in both arms; require cached_tokens==0 on every response",
            "global_settings_overrides": {
                "cache.enabled": False,
                "cache.hot_cache_max_size": "0",
                "cache.hot_cache_only": False,
            },
            "require_runtime_prefix_cache_disabled": True,
            "engram_rows": "Fresh process starts each arm/block. One identical excluded 16K warmup runs first. Retain all natural Engram row-cache state after warmup and through the measured sequence; no private clear or tensor mutation.",
            "row_cache_reporting": "Save public/read-only observer snapshots after load, after warmup, before/after each measured request and before shutdown when available. Record cache capacities, row-cache counters and their per-request deltas; missing telemetry is unavailable, never zero.",
            "os_cache": "Do not purge. OS file/page caches can persist across processes; counterbalanced process order reduces but does not eliminate this effect.",
            "interpretation": "Warm loaded-model performance with known row-cache history, not cold-load or cold-storage performance.",
        },
        "observations": {
            "per_process": [
                "exact wheel/source identity",
                "installed dependency/native-asset hashes",
                "effective safe settings before and after startup",
                "model/checkpoint/tokenizer identities",
                "hardware and memory limits",
                "runtime initial/after-warmup/final snapshots",
                "start/end times and exit status",
            ],
            "per_request": [
                "task_id, block, arm, order and preceding task IDs",
                "exact serialized request and SHA256",
                "HTTP status/errors",
                "client request-start, first-nonempty-content and stream-done monotonic timestamps",
                "all text/reasoning deltas, final finish reason and final usage",
                "actual prompt/completion/cached token counts",
                "server TTFT, prompt_eval_duration, generation_duration and token rates",
                "whole-output SHA256",
                "available row-cache and MTP counters before/after request",
            ],
            "ttft": "Client TTFT = elapsed from immediately before HTTP request send to receipt of the first SSE event with nonempty choices[].delta.content. Ignore role-only, empty content and usage-only events. Any unexpected reasoning/tool output is recorded and makes first-content interpretation explicit; do not silently count it as prose content.",
            "server_timings": "Preserve native usage values without relabeling as client timings. Native rates and durations are rounded by the server. Validate prompt_eval_duration and generation_duration are positive; missing required usage is an invalid measurement, not a zero.",
            "completion_latency": "Client stream-done latency includes transport and final usage; report as completed-answer latency only for pairwise identical text and ordinary stop in both arms. Length-stopped requests remain in TTFT and throughput tables but have no completed-answer latency claim.",
        },
        "validity": {
            "retain_every_attempt": True,
            "no_selective_retries_or_case_replacement": True,
            "required_measured_pairs": 16,
            "all_eight_workloads_in_each_block_must_match_text_and_finish_reason": True,
            "require_prompt_tokens_match_fixture_and_pair": True,
            "require_completion_tokens_match_within_pair": True,
            "require_cached_tokens_zero": True,
            "require_no_errors_and_complete_sse_usage": True,
            "require_versions_and_sources_match_frozen_plan": True,
            "require_prefix_cache_disabled_in_effective_config": True,
            "across_block_text_variation": "Report explicitly. Within-arm changes across deterministic repeats trigger investigation; do not silently treat them as timing noise.",
            "failure_policy": "Preserve failed or divergent observations, mark the paired qualification incomplete/failed, and do not present a successful overall speed claim. A rerun requires a documented reason and a new immutable run identity.",
        },
        "analysis": {
            "primary_metrics": [
                "client first-content TTFT in seconds",
                "native evaluated-prompt throughput in tokens/second",
                "native generation throughput in tokens/second",
            ],
            "secondary_metrics": [
                "native server TTFT",
                "native prompt/generation durations",
                "client stream-done latency only on identical nontruncated pairs",
            ],
            "pairing": "Compare same workload and block; show all16 pairwise results before any summary. For TTFT/latency use baseline/candidate; for rates use candidate/baseline, so values above1 favor the candidate.",
            "aggregation": "For each metric compute the geometric mean of the two block ratios within each case, then equal-case geometric means separately for synthetic1K (2cases), natural16K (4cases), natural64K (2cases), natural-all (6cases), and optionally all8 (explicitly labeled mixed). Show both block ratios and min/max across cases. Never weight by repetitions as if32 independent workloads existed.",
            "completion_latency_denominator": "Show exact eligible task IDs and completed-pair count. Never combine completed and length-stopped answers, and never call time-to512tokens time-to-complete-answer.",
            "no_quality_inference": "These workloads measure performance. Identical outputs establish paired comparability for these requests, not general model quality. Cite the separate618-task gate for broader bounded quality evidence.",
            "uncertainty": "Two temporal blocks on one device support a reproducible bounded comparison, not population significance, p95/p99 latency, broad confidence intervals or general hardware claims.",
            "historical_separation": "Do not transfer or combine the historical20.06% direct-inference result. This run uses ordinary EOS, greedy sampling, public streaming HTTP and a different final build; it stands on its own.",
        },
        "review_items_before_freeze": [
            "Bind final candidate and quality report identities",
            "Bind exact performance client/server observer source",
            "Confirm normal cache override is applied after quality configure and saved before server startup",
            "Freeze all payload/config/harness hashes and ensure no mutable source runtime is used",
        ],
        "builder_sha256": sha(Path(__file__).read_bytes()),
    }
    write_json(HERE / "plan-draft-01.json", plan)
    assert_cpu_only()
    print(
        json.dumps(
            {
                "status": plan["status"],
                "requests": len(jobs),
                "measured": 8,
                "exact_historical_measured_token_streams": sum(
                    p["historical_token_stream_reconstructed_exactly"]
                    for p in provenance
                    if p["historical_id"] != WARMUP
                ),
                "actual_prompt_tokens": {
                    p["task_id"]: p["actual_http_prompt_tokens"] for p in provenance
                },
                "plan_sha256": sha((HERE / "plan-draft-01.json").read_bytes()),
            }
        )
    )


if __name__ == "__main__":
    main()
