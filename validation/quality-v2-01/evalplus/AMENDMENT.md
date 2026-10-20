# Seeded evaluation amendment (02)

This is a separate evaluator protocol for fresh comparison of the original baseline response file and the later V2 candidate response file. Original evaluator source and results remain byte-for-byte unchanged under `../evalplus/`. Do not mix scores from the two protocols as though they used one evaluator.

The original adapter launched workers with `-I`, which ignores `PYTHONHASHSEED`. The amended adapter launches them with `-s -P -B` in the same explicit environment whitelist, adds `PYTHONNOUSERSITE=1`, and keeps `PYTHONPATH` absent. Thus user site packages and script/current-directory path injection are disabled while `PYTHONHASHSEED=0` actually applies. The same Seatbelt restrictions, limited read paths, filesystem/network/fork restrictions, resource limits, and monitored RSS policy remain in place.

Each worker seeds Python's shared `random` generator and NumPy's legacy shared generator to zero immediately before canonical execution and again before candidate execution. A repeated trusted probe inside separate Seatbelt workers checks hash values, shared random samples, startup flags, and absence of user path injection. This reduces process-to-process variation; it does not override independently created RNG objects, explicit user reseeding, entropy reads, clock reads, or every possible nondeterministic program.

Dataset releases, upstream EvalPlus commit, prompt/extraction rules, official sanitizer and special oracles, full task counts, test time limits, output budgets, and pass/fail aggregation are unchanged. The known upstream HumanEval/32 oracle bookkeeping issue and MBPP/630 generator sanitation issue remain in the canonical sweep and both model denominators. No model answer is repaired. Fresh baseline and V2 candidate evaluations must use this exact same amended adapter, with all 542 coding tasks retained.

The pinned upstream checkout, datasets, and evaluator environment are read from `../evalplus/{source,datasets,env}`. `adapter-source/` contains the separate amended grader and an unchanged exporter. `requirements-resolved.txt` is copied from the original environment. The amended source must be frozen before its probes, controls, canonical sweep, and model grading begin; metadata and receipts bind the exact bytes. The original `../evalplus/PROTOCOL.md` provides the underlying isolation protocol, licenses, and download pins, with this file superseding its interpreter/seed behavior only.

Validation outputs belong only in this directory. Repeat startup/isolation probes, the 12 fixed negative controls, the complete canonical sweep, and fresh baseline grading precede the V2 candidate grading. A validation receipt will state the actual results; this amendment document alone is not proof that a run completed. Raw exact-output parity, when measured by the parent workflow, remains independent evidence from test execution.

Example command for fresh original-baseline grading after controls:

```sh
work/flash1017-quality-01/evalplus/env/bin/python -I -B work/flash1017-quality-01/evalplus-deterministic-02/adapter-source/flash1017_evalplus_grade.py --source work/flash1017-quality-01/evalplus/source --datasets work/flash1017-quality-01/evalplus/datasets --work work/flash1017-quality-01/evalplus-deterministic-02/sandbox --samples work/flash1017-quality-01/baseline-02/responses.jsonl --output work/flash1017-quality-01/evalplus-deterministic-02/baseline-grades.jsonl
```

The supervisor can still use `-I`: the deliberately different child flags are set inside the amended adapter. Run the V2 candidate afterward with its actual saved response path and a distinct output filename. Outputs refuse overwrite. No model is loaded by this workflow.
