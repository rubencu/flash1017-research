# Focused oMLX contributions

This is a proposed contribution portfolio, not a set of submitted or merge-ready PRs. The complete research configuration is valuable independently of whether every component is accepted upstream.

Source review on **18 October 2026** found oMLX main at [`165be1ff`](https://github.com/jundot/omlx/tree/165be1ff5933a020f2f5a7544e1e95a2797f777f). The completed overall benchmark used installed **0.7.1rc1**, not that tree. Its result cannot substitute for current-main validation of extracted patches.

## First candidate: BF16 vocabulary weight reuse for rows 2–5

**Proposed change:** integrate reuse of one BF16 weight read across multiple verification rows into the native projection function, preserving its existing per-row reduction and unsupported-shape/device fallbacks. Keep this independent of reference rounding, confidence scheduling, replay and cache changes.

This is a credited adaptation of an established technique. A directly analogous [tacos8me implementation](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py) predates the exact-1017 work. Merged oMLX [#4475](https://github.com/jundot/omlx/pull/4475) handles related sharing in an **8-bit** path. Main's lack of this BF16 implementation is an integration opportunity, not evidence of invention.

**Evidence available:** the numerical audit for widths two through five matched 167,934,720 logits against the original shape-matched kernel. A historical fixed-depth-two study observed 7.18% and 7.51% higher generation rates at 1K and 16K, with 10,240 matching output positions over 20 paired requests. Both sides already used local reference rounding. That study is not an isolated current-main result across all supported widths.

**Prepared locally:** the [one-file patch packet](../upstream/bf16-rows2-5/README.md) contains the pinned base, production-function adaptation, draft PR and a finite validation checklist. Patch application, Python syntax and source-preservation checks pass without executing MLX.

**Before submission:** build pinned main and its patched counterpart; exercise the actual projection entry point, supported widths and fallbacks; run relevant and default project tests; and perform a bounded paired exact-model check. The packet is not merge-ready and its historical evidence is not a benchmark of the extracted patch.

**Keep six rows separate.** The local complete profile extends reuse to six rows, replacing the old matrix-multiply fallback. Its logits differ from that fallback even though the saved audit observed no argmax mismatch and matched the original single-row arithmetic. The unchanged-arithmetic claim for rows two through five does not cover six.

## Conditional compatibility candidate: attention reference rounding

**Proposed change:** an explicit option for the documented rounding boundary after inverse RoPE and before the attention output projection, in target and draft paths, with configuration and loading support.

Credit DeepSeek and [gnahzhang's documented exact-1017 reference behavior](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX/blob/c98bd97baca052f484c1d6ea6b6e8a8eced404ef/README.md). This carries **no speed or general quality-improvement claim**. The local coding checks were mixed.

**Before submission:** demonstrate the intended reference contract with a small fail-before/pass-after test through production target and draft code; verify disabled-mode compatibility and configuration loading; and perform a representative exact-checkpoint check. If the contract remains uncertain, retain experimental status instead of asserting an upstream defect.

## Later candidates

| Area | Coherent scope | Remaining evidence or coordination |
|---|---|---|
| Confidence-prefix selection | DSpark causal stopping, frozen costs/calibration, scoped request ownership and supported-mode fallback | Compare native adaptive and confidence selection with the same arithmetic. Preserve context-copy behavior and address the tested head6 dependency. Hardware-specific calibration is not a universal default. |
| Dependency replay | Layout/dependency guards, numerical dispatch preservation, physical draft-state priming and lifecycle behavior | Keep the required numerical repairs with the feature. Validate native fallback and useful wide-chunk activation through production code. The historical 40.3% result combines replay and cache. |
| Prefetch configuration | Bounded, explicit admission configuration for the existing reader | Coordinate with wide-step [#3949](https://github.com/jundot/omlx/pull/3949). Do not propose an unconditional global 16-to-64 MiB change as universally faster. |
| HotRows integration | Bounded model-owned cache, reader coordination, fallback and cleanup | Credit the reused implementation and coordinate with storage proposal [#4611](https://github.com/jundot/omlx/issues/4611). Test ownership, memory limits, pending work, cancellation and unload. |

Confidence scheduling follows [DeepSeek DSpark](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2); earlier community cost policies also exist. Replay and partition-preservation techniques have prior implementations. The local work supplies concrete adaptation and validation, not new general algorithms. [ATTRIBUTION.md](ATTRIBUTION.md) records the distinctions.

Replay-induced numerical changes are not independently established bugs in ordinary upstream inference. A useful PR keeps the fixes needed for the new path together. Similarly, the isolated service's single-active-request policy does not prove arbitrary batching or multi-model support.

## Upstream state to preserve

At the dated review, [#3949](https://github.com/jundot/omlx/pull/3949) remained open with conflicts, and [#4611](https://github.com/jundot/omlx/issues/4611) remained an open overlapping proposal. Open [#4648](https://github.com/jundot/omlx/pull/4648) touches shared MTP/controller code; it is not our confidence policy, but requires overlap review. Merged [#4363](https://github.com/jundot/omlx/pull/4363) already provides context-copy and request-local DSpark priming. Merged [#4601](https://github.com/jundot/omlx/pull/4601) changed native kernel namespaces/build layout, so Python-only overlays are insufficient for a current-main build.

oMLX already has approximate CED via [#3607](https://github.com/jundot/omlx/pull/3607). The full-depth reference used here has a different contract; no performance superiority over that CED mode was measured.

The [contribution guide](https://github.com/jundot/omlx/blob/165be1ff5933a020f2f5a7544e1e95a2797f777f/docs/CONTRIBUTING.md) welcomes hardware-specific improvements and calls for focused tests, relevant main-based performance measurements and retained credits. New subsystems and large first contributions require scope discussion. Model-limited evidence is useful when the implementation and claims are correspondingly scoped.

No separate MLX-core defect, independently validated primitive or finished MLX-LM port has been established. Those projects are not immediate PR destinations merely because this work uses MLX.

The remaining work is bounded extraction and verification of selected changes. This preview neither starts another optimization sweep nor promises acceptance upstream. The [complete report](../evidence/overall/review/report.txt) and [source archive](../source/) support review; they do not make these proposed patches merge-ready.
