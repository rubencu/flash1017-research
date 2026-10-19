# Flash1017 for oMLX — research preview

**18 October 2026. Source and evidence release; standalone installation is not yet verified.** This package records an experimental oMLX configuration for one exact DeepSeek-V4.1-Flash-1017 checkpoint. It contains no new model weights or quantization.

## Included work

The accepted configuration includes BF16 vocabulary weight reuse through six rows; optional target/draft reference rounding; causal confidence-prefix scheduling and its frozen calibration; 8K chunks and 64 MiB native prefetch admission; dependency replay and numerical repairs; three-stage DSpark priming; bounded HotRows integration; and the supporting ownership, fallback, cleanup and activation checks. The [source archive](../source/) preserves the implementation and its required supporting definitions.

The [analysis](../analysis/) and [overall report](../evidence/overall/review/report.txt) distinguish saved-data recalculation from real-model execution. Historical component and negative results remain separately labeled. Credits and code provenance are in [ATTRIBUTION.md](ATTRIBUTION.md).

## Measured result

On one 80-GPU-core M5 Ultra with 256 GiB memory, the complete configuration reduced first-token time **20.06%**, from **19.554 to 15.631 seconds** by equal-case geometric mean, versus untouched installed **oMLX 0.7.1rc1 with native adaptive MTP enabled and 2K chunks**. The suite used eight cases, two process blocks per arm and 48 measured requests across three arms. It is a warm-runtime comparison; loading and HTTP transport are excluded.

Material limits:

- The 64K CPython case regressed **2.32%** in first-token time.
- Sampled peak footprint increased **208.35 → 216.10 GiB**. These are sampled peaks, not memory guarantees.
- The fixed-512 useful-answer qualification failed: **nine early EOS observations, three per arm**. Forced post-EOS timings are descriptive.
- The **20.21%** time-to-token-128 reduction is a **post-hoc** analysis within the original runs.
- Outputs differ. Answer-quality equivalence and faster equivalent task completion have not been established.
- The denominator is installed 0.7.1rc1, **not current upstream main**. Two blocks on one machine do not establish performance on other hardware or workloads.

The earlier **40.3%** prompt-processing reduction is a different replay-plus-cache study against an already modified reference. Do not combine the percentages.

## Scope and status

The local serving profile targets only [mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab), revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, with one active request and a **131,072-token total context limit**. Approximate CED is disabled. The timing suite covers inputs through 64K.

The implementation still reflects a laboratory environment: some early hooks are process-global, and launch/build dependencies require extraction before ordinary installation. No verified binary installer, arbitrary concurrent multi-model support, smaller-Mac support, current-main benchmark, external replication or optimized 1M serving is claimed. The failed long-serving gate is preserved.

Rejected or unqualified variants—including the larger retained cache, envelope padding and alternative gather/radix experiments—are evidence, not enabled release features.

Read the [article](ARTICLE.md) for the engineering explanation and [UPSTREAM.md](UPSTREAM.md) for the proposed contribution sequence. This preview is useful for source review, saved-result verification and adapting individual changes; a runnable distribution is a separate packaging milestone.
