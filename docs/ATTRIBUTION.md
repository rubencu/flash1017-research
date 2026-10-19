# Attribution and contributions

This project integrates and evaluates execution changes for a specific
DeepSeek-V4.1-Flash-1017 checkpoint in oMLX. Its contributions include numerical
and draft-state repairs, model-scoped runtime integration, resource cleanup,
packaging, and measurements tied to exact source and configuration records.
The model, quantization and general optimization techniques come from the
foundations and related work identified here.

## Foundations

- **DeepSeek** supplies [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  and the [DSpark speculative-decoding method](https://arxiv.org/html/2607.05147v1).
- **mlx-community** supplies the unchanged
  [oQ2.5e-MTP checkpoint at revision `78456115`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab).
- **[Apple MLX](https://github.com/ml-explore/mlx)**,
  **[MLX-LM](https://github.com/ml-explore/mlx-lm)** and
  **[MLX-VLM](https://github.com/Blaizzy/mlx-vlm)** provide array and model infrastructure.
- **[oMLX](https://github.com/jundot/omlx)** provides loading, serving, scheduling,
  speculative execution, native kernels and storage infrastructure. The overall
  performance comparison uses installed release 0.7.1rc1 as its reference.

## Reused code, related implementations and project additions

| Component | Source or related work | Contribution in this project |
|---|---|---|
| BF16 vocabulary-weight reuse | Adaptation of oMLX's projection; analogous multi-row reuse in [tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py). | Kernel adaptation, six-row extension, integration and comparisons distinguishing rows 2–5 from the six-row fallback. |
| Confidence scheduling | [DSpark's causal stopping method](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2); related [confidence/cost policy](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/mtp.py). | Previous-token conditioning, per-position calibration, a zero-draft option and a fixed cost policy integrated into the selected runtime. |
| Dependency replay | Decoder-suffix and arithmetic-partition preservation in [antirez/ds4](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42408); related [MLX replay implementation](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/encoder_replay.py). | Adaptation to the pinned model, shape-sensitive numerical repairs, preservation of all three draft-stage projections, state checks, fallback and lifecycle integration. |
| HotRows cache | **Copied unchanged** from [tacos8me/m5-ultra at `fb233e2`](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/hot_rows.py), with its Apache-2.0 license and notice. | Model-owned adapter, bounded budgets, native-reader coordination, fallback, cleanup and measurements. |
| Engram prefetch | oMLX's existing selected-row reader and background executor. | Evaluation and integration of a larger prefetch admission budget with wider prompt chunks; instance-local budget handling in the fork. |
| Attention-output rounding | DeepSeek reference behavior and [gnahzhang's exact-model conversion/runtime documentation](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX/blob/c98bd97baca052f484c1d6ea6b6e8a8eced404ef/README.md). | Implementation and investigation of that boundary in the selected target and draft execution paths. |

These relationships distinguish a direct source copy, an adaptation and related
implementations. Similar implementations alone do not establish direct copying
or independent invention. The source-level notices and the
[component attribution accompanying the prototype](../archives/2026-10-18-preview/docs/ATTRIBUTION.md)
provide additional references and license details.

## Licenses and review provenance

Original additions and repository documentation use the root
[Apache-2.0 license](../LICENSE). Existing per-file MIT and Apache-2.0 licenses
remain applicable; the snapshots are not blanket relicensed. The
[fixture notices](../archives/2026-10-18-preview/evidence/overall/fixtures/NOTICE)
cover CPython, curl and Project Gutenberg material. Model weights and an
application distribution are not included in this research repository.

AI coding agents assisted implementation, investigation, testing, evidence
analysis and drafting. Additional local code paths and agents checked selected
sources and saved results. Those are local audits; external peer review and
independent-machine replication have not been performed.
