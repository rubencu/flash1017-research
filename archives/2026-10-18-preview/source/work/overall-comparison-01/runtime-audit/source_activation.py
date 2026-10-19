"""CPU-only observations of an already loaded comparison arm.

This module imports no inference package, loads no model, evaluates no tensor,
and changes no runtime function. Call only outside timed generation. It records
source identity and configured activation, not proof that every GPU branch ran.
The caller must retain per-request MTP and cone/cache counter evidence too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


def _digest(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _function(function):
    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    result = {"module": getattr(function, "__module__", None),
              "qualname": getattr(function, "__qualname__", None)}
    if code is not None:
        result.update(_digest(code.co_filename))
        result["line"] = code.co_firstlineno
    return result


def _below(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def capture(*, runtime_root, model, arm, profile, owner=None):
    """Return observations/checks; never infer successful execution from them.

    runtime_root is the selected *omlx package directory*, not its parent.
    arm is 'upstream' or 'candidate'. Profiles must explicitly provide mtp,
    depth, fixed, prefill_step, and ced_prefill. Candidate additionally supplies
    head_max_rows, draft_width, and engram_prefetch_bytes. A failed check is
    returned for the caller to persist before raising.
    """
    if arm not in ("upstream", "candidate"):
        raise ValueError("arm must be upstream or candidate")
    root = Path(runtime_root).resolve()
    prefix = "omlx.patches.deepseek_v41."
    required = [prefix + item for item in ("language", "storage", "head", "dspark", "config", "loading")]
    required.append("omlx.patches.mlx_lm_mtp.batch_generator")
    missing = [name for name in required if name not in sys.modules]
    if missing:
        raise RuntimeError("Run capture after native setup/load: " + ", ".join(missing))
    language, storage, head, dspark, config, loading, batch = [sys.modules[name] for name in required]
    target = model.language_model
    configuration = target._config
    sources, escapes = {}, []
    for name, module in sorted(tuple(sys.modules.items())):
        if name != "omlx" and not name.startswith("omlx."):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        sources[name] = _digest(path)
        if not _below(path, root):
            escapes.append({"module": name, "path": str(Path(path).resolve())})
    functions = {
        "head": _function(head.project_logits),
        "target_head": _function(language.project_logits),
        "draft_head": _function(dspark.project_logits),
        "target_call": _function(type(target).__call__),
        "target_prefill": _function(target._omlx_prefill),
        "draft_loop": _function(batch._dspark_next_drafts),
        "verify_loop": _function(batch._run_verify_cycle_chain),
        "loader": _function(loading.load),
    }
    experiment = getattr(target, "_flash1017_experiment", None)
    cone_owner = getattr(target, "_serving_cone_v3_owner", None)
    row_owner = getattr(model, "_flash1017_serving_row_cache", None)
    state = {
        "attention_rounding": bool(getattr(configuration, "attention_output_quantization", False)),
        "attention_rounding_field_present": hasattr(configuration, "attention_output_quantization"),
        "prefetch_bytes": storage.PREFETCH_BYTES,
        "ced_prefill": bool(configuration.ced_prefill),
        "mtp_enabled": bool(getattr(target, "_omlx_mtp_decode_enabled", False)),
        "dspark_enabled": bool(getattr(target, "_omlx_dspark_decode_enabled", False)),
        "mtp_depth": getattr(target, "_omlx_mtp_depth", None),
        "mtp_fixed": bool(getattr(target, "_omlx_mtp_depth_fixed", False)),
        "confidence_policy": experiment is not None and getattr(experiment, "policy", None) is not None,
        "draft_width": getattr(experiment, "draft_width", None),
        "cone_attached": cone_owner is not None,
        "row_cache_attached": row_owner is not None,
        "native_target_class": type(target) is language.LanguageModel,
        "prefill_step_requested": profile["prefill_step"],
    }
    checks = {
        "all_imported_omlx_sources_under_selected_root": not escapes,
        "head_aliases_agree": head.project_logits is language.project_logits is dspark.project_logits,
        "requested_mtp_enabled": state["mtp_enabled"] is profile["mtp"] and state["dspark_enabled"] is profile["mtp"],
        "requested_mtp_depth": state["mtp_depth"] == profile["depth"],
        "requested_fixed_policy": state["mtp_fixed"] is profile["fixed"],
        "requested_ced_mode": state["ced_prefill"] is profile["ced_prefill"],
    }
    if arm == "upstream":
        checks.update({
            "native_head_function": functions["head"].get("path") == str((root / "patches/deepseek_v41/head.py").resolve()),
            "no_local_rounding_field": not state["attention_rounding_field_present"],
            "native_prefetch_budget": state["prefetch_bytes"] == 16 * 2**20,
            "no_local_experiment_marker": not hasattr(target, "_flash1017_experiment"),
            "no_candidate_owners": owner is None and cone_owner is None and row_owner is None,
            "native_target_class": state["native_target_class"],
            "no_head_patch_marker": not hasattr(head, "_flash1017_original"),
        })
    else:
        closure = dict(zip(head.project_logits.__code__.co_freevars,
                           (cell.cell_contents for cell in (head.project_logits.__closure__ or ()))))
        state["head_max_rows"] = closure.get("max_rows")
        checks.update({
            "candidate_head_function": functions["head"].get("module") == "flash1017.head",
            "requested_head_width": state["head_max_rows"] == profile["head_max_rows"],
            "requested_attention_rounding": state["attention_rounding"] is profile["attention_rounding"],
            "requested_prefetch_budget": state["prefetch_bytes"] == profile["engram_prefetch_bytes"],
            "confidence_active": state["confidence_policy"] and state["draft_width"] == profile["draft_width"],
            "candidate_owners_match": owner is not None and cone_owner is owner.cone and row_owner is owner.rows,
            "candidate_enabled": owner is not None and owner.cone.enabled and owner.rows.enabled and not owner.closed,
        })
    return {"scope": "CPU source/configuration observations only; no tensor evaluation or execution-branch proof",
            "arm": arm, "runtime_root": str(root), "checks": checks, "passed": all(checks.values()),
            "state": state, "functions": functions, "sources": sources, "source_escapes": escapes}
