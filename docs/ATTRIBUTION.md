# Attribution and contributions

This project integrates and evaluates execution changes for a specific
DeepSeek-V4.1-Flash-1017 checkpoint in oMLX. The measured profile integrates
vocabulary-weight reuse, bounded lookup-row caching and prefetch admission,
with resource cleanup, packaging and evaluation tied to exact build identities.
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
  speculative execution, native kernels and storage infrastructure. The performance
  comparison uses the public `v0.7.1rc1` wheel as its reference.

## Used in the measured profile

| Component | Source or related work | Contribution in this project |
|---|---|---|
| BF16 vocabulary-weight reuse | Adaptation of oMLX's projection; analogous multi-row reuse in [tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py). | Integration for rows 2–5 with stock per-row arithmetic, dispatch limits and exact Metal parity checks. |
| HotRows cache | **Copied unchanged** from [tacos8me/m5-ultra at `fb233e2`](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/hot_rows.py), with its Apache-2.0 license and notice. | Model-owned adapter, bounded budgets, native-reader coordination, fallback, cleanup and evaluation. |
| Engram prefetch | oMLX's selected-row reader and background executor. | Instance-local admission budget, a 64 MiB setting, lifecycle integration and evaluation with stock prompt chunks. |

The contribution is implementation and verification for the identified runtime,
not invention of weight reuse, caching, prefetch or speculative decoding.
A related implementation is distinct from a direct source copy; the HotRows
copy is explicitly identified above and in its source notice.

## Credits for other included source

The source archive and inactive runtime modules include work on additional
execution paths. They are not part of the measured profile or its headline:

- Decoder-suffix replay and arithmetic-partition preservation relate to
  [antirez/ds4](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42408)
  and [tacos8me's MLX replay](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/encoder_replay.py).
- Confidence scheduling follows [DSpark's causal stopping method](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2),
  with related [confidence/cost policy code](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/mtp.py).
- Attention-output rounding was informed by DeepSeek reference behavior and
  [gnahzhang's conversion/runtime documentation](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX/blob/c98bd97baca052f484c1d6ea6b6e8a8eced404ef/README.md).

The [source attribution record](../archives/2026-10-18-preview/docs/ATTRIBUTION.md)
provides component details and license references. Per-file notices remain
applicable to both active and inactive source.

## Licenses and authorship

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
