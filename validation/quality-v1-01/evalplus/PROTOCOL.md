# Flash1017 installed-build coding evaluation

This is a local macOS execution adapter for the pinned EvalPlus tests, not a claim that the unmodified EvalPlus CLI was run. The supervisor never executes generated or reference code. Both are executed only inside a fresh OS sandbox for each task and test phase. The two saved generation arms use the same adapter and environment. No answer is manually repaired or dropped.

## Frozen inputs

EvalPlus v0.3.1 is pinned at `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` in `source/`. The runner verifies that commit, the tracked `evalplus/` source is unchanged, and no untracked Python source is present there. Dataset identities and download URLs are in `datasets/identities.json`.

| Suite | Release | Tasks | Original inputs | Additional inputs |
| --- | --- | ---: | ---: | ---: |
| HumanEval+ | v0.1.10 | 164 | 1,570 | 122,683 |
| MBPP+ | v0.2.0 | 378 | 1,174 | 39,841 |

These are the full pinned releases, without Mini or NoExtreme filtering. `requests.jsonl` contains all 542 tasks and no reference solutions or hidden test inputs. Its SHA256 is `45f5ab9e058993b8ca2876337745adc4cfedc1bfffbb7d060484422288ff3a06`.

The exporter reproduces the official zero-shot chat system message and user instruction from [the v0.3.1 OpenAI provider](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/provider/openai.py) and [message construction](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/gen/util/openai_request.py). Generation was one answer per task, temperature 0, top_p 1, seed 431017, thinking disabled, 2048 output tokens, no assistant prefill, and ordinary EOS. The 2048-token/top_p=1 settings are this paired experiment's frozen settings, not identical to all upstream defaults or a leaderboard submission.

## Scoring

The worker imports the pinned upstream sanitizer, canonical executor, MBPP input deserializer, `unsafe_execute`, special oracles, and pass/fail constants. The source is not patched. The full answer is passed through the official sanitizer with the task entry point. Sanitizer exceptions are failed answers. The original response bytes remain in the saved arm file, and each successful worker record stores raw and sanitized SHA256 values plus the sanitized solution.

Each phase first computes its expected values and reference timings using upstream `trusted_exec`, inside the sandbox. Candidate checking uses the unchanged upstream `unsafe_execute` and special numerical/set/not-none oracles. Original and additional phases are both run. `base_pass` means original tests passed. `plus_pass` requires both phases to pass. Pass@1 is the fraction of the full suite with `plus_pass=true`; there is exactly one completion per task. Timeouts, memory/output kills, empty answers, malformed answers, and generation truncations are retained. An infrastructure failure stops the run rather than silently becoming a model error.

The adapter replaces multiprocessing shared transport with local holders and a parent-supervised process, replaces the temporary-directory context with a precreated private directory, and sets `tempfile.tempdir` for import-only dependencies. It preserves the upstream Python reliability guard except its unsupported address-space/data memory limit call. Per-input time is `max(1 second, 4 × measured canonical time)`; the candidate phase wall budget is `min(60 seconds, sum(per-input limits)) + 2 seconds`, matching upstream fast-check join allowances. The canonical/setup phase has a separate 180-second wall cap. These choices follow the pinned [execution protocol](https://github.com/evalplus/evalplus/blob/v0.3.1/docs/execution.md) and [checker implementation](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/__init__.py).

The two-process concurrency limit bounds monitored worker RSS to 8 GiB in aggregate before polling overshoot. Typical workers are much smaller. Reference timings are measured locally for each phase and arm; this is not a performance benchmark. Concurrent diagnostic model work can affect timings. The one-second minimum and matched procedure are held fixed, without tuning to model results.

Interpreter isolation uses `-I`, which ignores `PYTHONHASHSEED` even though the child environment contains that variable. Python string hashing is therefore randomized per process; upstream does not seed arbitrary solution randomness either. `hash-seed-observation.json` confirms this with trusted startup probes. Identical generated text is not a general proof of identical future test verdicts. If a later result reuses a saved verdict, it must be labeled as derived from that saved evaluation, not a fresh execution. No seed or scoring changes were made after inspecting answers.

## Isolation and limits

`/usr/bin/sandbox-exec` applies a deny-by-default Seatbelt profile. Reads are limited to interpreter/dependency libraries, pinned evaluator code, the two helper scripts, and that worker's input directory; root directory listing and general metadata reads are allowed for runtime startup. Home/workspace file contents outside those paths are denied. Filesystem writes, network access, forks, and foreign executable launches are denied. Child environment variables are replaced with a small fixed environment. No Docker/VM is used.

The worker sets hard limits of 180 CPU seconds, 1 MiB regular-file output, 64 file descriptors, and zero core size before reference/candidate execution. The parent bounds captured stdout to 1 MiB and enforces phase wall deadlines. Stderr is limited by the file-size limit and retained in bounded diagnostics on failure.

On this macOS version, lowering RLIMIT_AS or RLIMIT_DATA from unlimited to 4 GiB fails. The parent therefore samples RSS every 20 ms and kills workers above 4 GiB. **This is a monitored RSS threshold, not a hard allocation cap; allocations can overshoot between samples.** The memory probe records the observed overshoot. It is not an adversarially hardened judging service: public tests and their references exist in the worker's readable input, and generated code shares a process with the oracle. The OS sandbox protects the host within its tested rules; no claim is made that generated code cannot intentionally manipulate its own scoring process.

## Validation and known pinned behavior

`isolation-probe-v2.json` records denied outside reads/writes, scratch writes, sockets, forks, and foreign executable launches; CPU-limit, RSS-watchdog, wall-watchdog, and output-watchdog probes also pass. `negative-controls-v2.json` contains 12 rejection checks: wrong return value, malformed code, and infinite loop, each checked on both phases of fixed HumanEval/0 and Mbpp/2 tasks. `canonical-grades-v2.jsonl` is the full canonical-answer sweep. Earlier smoke attempts and the intentionally interrupted first sweep are retained as diagnostic history, not final results.

The complete canonical sweep passes 540/542 tasks. The canonical `HumanEval/32` (`find_zero`) is rejected by the pinned evaluator: its special oracle takes `continue` before updating progress/details, then the official final completeness check marks the phase failed. `Mbpp/630` is rejected because the official sanitizer removes its generator helper, which has `yield` but no `return` statement. Expected-value execution of that canonical source succeeds before sanitation. A separately labeled synthetic positive control adding an explicit return after generator exhaustion preserves the helper and passes both phases, confirming the cause; it is not a repaired model answer or included benchmark score. Both upstream behaviors remain unchanged, and both tasks remain in the full denominators. These are limitations of the pinned scoring protocol, not proof of model errors on those items. See `validation-receipt.json` and `generator-sanitizer-control.json`.

## Running from this workspace

The commands below use the isolated evaluator environment, not the model environment. `requirements-resolved.txt` records all installed versions. Python is 3.12.15 on macOS arm64. The source checkout and dataset files must match the pins above. Never invoke generated code directly or run an unsandboxed fallback.

For a fresh local copy, first place the two adapter scripts under the indicated research `tools/` directory and this protocol's `requirements-resolved.txt` in the evaluator directory. With CPython 3.12.15 available, prepare the environment and public inputs as follows. The exporter and grader verify the uncompressed dataset hashes before use. This setup does not install a model runtime or launch a model.

```sh
mkdir -p work/flash1017-quality-01/evalplus/datasets
python3.12 -m venv work/flash1017-quality-01/evalplus/env
work/flash1017-quality-01/evalplus/env/bin/python -m pip install -r work/flash1017-quality-01/evalplus/requirements-resolved.txt
git clone --branch v0.3.1 https://github.com/evalplus/evalplus.git work/flash1017-quality-01/evalplus/source
git -C work/flash1017-quality-01/evalplus/source checkout --detach e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2
curl --fail --location https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz -o work/flash1017-quality-01/evalplus/datasets/HumanEvalPlus-v0.1.10.jsonl.gz
curl --fail --location https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz -o work/flash1017-quality-01/evalplus/datasets/MbppPlus-v0.2.0.jsonl.gz
gzip -dc work/flash1017-quality-01/evalplus/datasets/HumanEvalPlus-v0.1.10.jsonl.gz > work/flash1017-quality-01/evalplus/datasets/HumanEvalPlus-v0.1.10.jsonl
gzip -dc work/flash1017-quality-01/evalplus/datasets/MbppPlus-v0.2.0.jsonl.gz > work/flash1017-quality-01/evalplus/datasets/MbppPlus-v0.2.0.jsonl
work/flash1017-quality-01/evalplus/env/bin/python -I -B repositories/flash1017-research/tools/flash1017_evalplus_export.py --datasets work/flash1017-quality-01/evalplus/datasets --output work/flash1017-quality-01/evalplus/requests.jsonl
```

Generation is a separate root-owned workflow. To grade already saved outputs, run:

```sh
work/flash1017-quality-01/evalplus/env/bin/python -I -B repositories/flash1017-research/tools/flash1017_evalplus_grade.py --source work/flash1017-quality-01/evalplus/source --work work/flash1017-quality-01/evalplus/sandbox --probe

work/flash1017-quality-01/evalplus/env/bin/python -I -B repositories/flash1017-research/tools/flash1017_evalplus_grade.py --source work/flash1017-quality-01/evalplus/source --datasets work/flash1017-quality-01/evalplus/datasets --work work/flash1017-quality-01/evalplus/sandbox --controls --output work/flash1017-quality-01/evalplus/negative-controls-new.json

work/flash1017-quality-01/evalplus/env/bin/python -I -B repositories/flash1017-research/tools/flash1017_evalplus_grade.py --source work/flash1017-quality-01/evalplus/source --datasets work/flash1017-quality-01/evalplus/datasets --work work/flash1017-quality-01/evalplus/sandbox --canonical --output work/flash1017-quality-01/evalplus/canonical-new.jsonl

work/flash1017-quality-01/evalplus/env/bin/python -I -B repositories/flash1017-research/tools/flash1017_evalplus_grade.py --source work/flash1017-quality-01/evalplus/source --datasets work/flash1017-quality-01/evalplus/datasets --work work/flash1017-quality-01/evalplus/sandbox --samples work/flash1017-quality-01/baseline-02/responses.jsonl --output work/flash1017-quality-01/evalplus/baseline-grades.jsonl

work/flash1017-quality-01/evalplus/env/bin/python -I -B repositories/flash1017-research/tools/flash1017_evalplus_grade.py --source work/flash1017-quality-01/evalplus/source --datasets work/flash1017-quality-01/evalplus/datasets --work work/flash1017-quality-01/evalplus/sandbox --samples work/flash1017-quality-01/candidate-02/responses.jsonl --output work/flash1017-quality-01/evalplus/candidate-grades.jsonl
```

Outputs refuse overwrite. Both model-arm commands must be run sequentially to keep at most two workers total. Each output sidecar binds the exact helper SHA, input response SHA, dataset pins, task IDs, Python version, and limits. The runner accepts `text` or `raw_completion`, ignores only IDs from other declared suites, and rejects missing or duplicate coding samples.

## Attribution and redistribution

EvalPlus is Apache-2.0, with OpenAI MIT notices for the evaluator code inherited from HumanEval. HumanEval+ is Apache-2.0 with the HumanEval MIT attribution; MBPP+ release data is Apache-2.0. Downloaded full license text and source URLs/hashes are in `licenses/`. Preserve these with any copied datasets or evaluator source. Dataset release sources are [HumanEval+ v0.1.10](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10) and [MBPP+ v0.2.0](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0). Nothing in this directory has been published.
