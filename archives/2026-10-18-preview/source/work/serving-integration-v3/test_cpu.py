"""Evidence contracts; deliberately imports no inference framework."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qualification import result_comparison, validate_plan, await_drained_future


def result():
    return {"input_tokens_sha256":"input", "seed":12, "token_ids":[4,5],
            "text":"text", "finish_reason":"length", "generated_tokens":2,
            "qualification":{"requested_output_tokens":2,"activation":{"passed":True}},
            "fixed_length_valid":True,"first_eos_position":None,
            "events":[{"verified":2,"accepted":1,"cycle_ms":9.0,"confidence_logits":[1.,2.]}],
            "mtp_stats":[{"cycles":1,"accepts":1,"backbone_ms":10.0}]}


def test_timing_only_differences_are_not_functional():
    a,b=result(),result()
    b["events"][0]["cycle_ms"]=13
    b["mtp_stats"][0]["backbone_ms"]=20
    assert result_comparison(a,b)["passed"]


@pytest.mark.parametrize("change", ["input","seed","ids","text","finish","confidence","acceptance","length","eos","bypass"])
def test_functional_mismatch_is_not_hidden(change):
    a,b=result(),result()
    if change == "input": b["input_tokens_sha256"]="wrong"
    elif change == "seed": b["seed"]+=1
    elif change == "ids": b["token_ids"][0]+=1
    elif change == "text": b["text"]="wrong"
    elif change == "finish": b["finish_reason"]="stop"
    elif change == "confidence": b["events"][0]["confidence_logits"][0]+=0.01
    elif change == "acceptance": b["mtp_stats"][0]["accepts"]+=1
    elif change == "length": b["generated_tokens"]-=1
    elif change == "eos": b["first_eos_position"]=0; b["fixed_length_valid"]=False
    elif change == "bypass": b["qualification"]["activation"]["passed"]=False
    assert not result_comparison(a,b)["passed"]


def plan():
    ids=[1,2,3]
    return {"profiles":{"native64":{"setting":1},"fastpath64":{"setting":1}},
            "prompts":{"p":{"tokens":ids,"input_tokens_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest()}},
            "jobs":[{"profile":"native64","prompt":"p","seed":1},
                    {"profile":"fastpath64","prompt":"p","seed":1}]}


def test_valid_frozen_pair():
    validate_plan(plan(),{"runtime":{"setting":1}})


@pytest.mark.parametrize("change", ["seed","profile","token","empty","extra_job"])
def test_plan_rejects_unmatched_inputs(change):
    p=plan()
    if change == "seed": p["jobs"][1]["seed"]+=1
    elif change == "profile": p["profiles"]["fastpath64"]["setting"]+=1
    elif change == "token": p["prompts"]["p"]["tokens"][0]+=1
    elif change == "empty": p["prompts"]["p"]["tokens"]=[]
    elif change == "extra_job": p["jobs"].append(copy.deepcopy(p["jobs"][0]))
    with pytest.raises(ValueError):
        validate_plan(p,{"runtime":{"setting":1}})


def test_import_has_not_loaded_inference():
    assert not any(name == "mlx" or name.startswith("mlx.") or name == "omlx" or name.startswith("omlx.")
                   for name in sys.modules)


def test_interrupted_wait_joins_worker_before_reraising():
    class InterruptedFuture:
        calls=0
        def result(self, timeout=None):
            self.calls+=1
            if self.calls == 1:
                raise KeyboardInterrupt("waiting thread interrupted")
            return "worker completed"
        def done(self):
            return self.calls >= 2
    future=InterruptedFuture()
    with pytest.raises(KeyboardInterrupt, match="waiting thread interrupted"):
        await_drained_future(future)
    assert future.done()


def test_cleanup_attempts_cone_after_row_close_error():
    from runtime import Candidate
    from types import SimpleNamespace
    owner=Candidate.__new__(Candidate)
    owner.closed=False
    owner.model=owner.target=object()
    owner.original_parameters=()
    owner.rows=SimpleNamespace(status="active")
    owner.cone=SimpleNamespace(closed=False)
    def row_close():
        owner.rows.status="detached"
        raise RuntimeError("retained read failure")
    def cone_close():
        owner.cone.closed=True
    owner.rows.close=row_close
    owner.cone.close=cone_close
    owner.verify=lambda:None
    with pytest.raises(RuntimeError,match="retained read failure"):
        owner.close()
    assert owner.closed and owner.cone.closed and owner.model is None and owner.target is None
