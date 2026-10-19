# Historical evidence: scope and omissions

The overall headline is supported by `../overall/`. This directory preserves earlier component observations and closed investigations. They are separate experiments, often using already modified comparators, and their gains are never added or multiplied.

`results/` includes:

- `heldout-01.jsonl` and summaries: the head, native-adaptive, confidence and early installed-stock cohorts, with their recorded profiles.
- `head-full-numerics-01.json`, `head-single-row-numerics-01.json` and `numerical-head-01.jsonl`: historical numerical findings. Widths 2–5 match the old specialized implementation; width six differs from the old matrix-multiply fallback while matching the single-row reference.
- `confidence-16k-*`, `cost6-01.jsonl` and the fitted policy: calibration and the 16K scheduling follow-up. Calibration inputs are separate from the held-out performance suite.
- `quality-*` and `deployment-quality-*`: small local quality checks, including failures. These do not establish general quality equivalence.
- `engram-prefetch-pilot-*` and `prefetch64-integration-*`: the small prefetch experiment and integration observations.
- `cone-shapes-01.json` and `cone-dspark-stable-01.json`: numerical diagnosis records. Large associated activation/state arrays are omitted.
- `serving-v3-performance-*`: replay plus HotRows against the already patched native64 reference. Its approximately 40.3% prompt-processing result is **not** an overall-versus-stock result.
- `serving-v3-boundary-*` and candidate HTTP/lifecycle records: finite historical model-state and serving checks. Full large state arrays are omitted; these copies alone cannot repeat the state comparison.

`plans/` holds the earlier laboratory's saved component plans. Their paths and execution environments are historical. The new overall study's complete exact input vectors and schedule are in `../overall/plan-01.json`.

`closed-studies/` preserves selected reports and receipts for larger-cache, envelope, whole-request and long-context investigations. These are not a full raw archive of those experiments. In particular the final long-HTTP recovery gate failed. Do not infer optimized 1M serving support from offline equality or an earlier interim status.

The selected reports came from a previously frozen local review bundle. A local independent audit means another local code path or agent checked saved evidence; it does not mean external peer review or replication on another machine. Original receipts may reference omitted files and original hashes. `../../EVIDENCE-INVENTORY.json` describes every copied file and path-only transformation.

Intentionally omitted: model weights, copied application/native binaries, full environment directories, live settings/state, large historical activation arrays and unselected raw traces. The complete new overall-comparison raw records and exact input tokens are included. The portable analyzer recalculates the overall result; historical scripts and reports are supporting records, not promises that every old experiment reruns from this archive.
