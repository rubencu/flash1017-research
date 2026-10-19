# Attribution and contribution map

This project adapts and evaluates existing techniques for one DeepSeek-V4.1-Flash-1017 configuration in oMLX. It did not create the model, quantization, MLX framework or oMLX serving engine. Algorithmic ideas and related implementations remain credited even when the local source differs.

This map supplements the source files' licenses and notices; it does not replace or relicense them. The original historical files remain unchanged. Distributed copies may have filesystem-prefix normalization, as recorded with their original and packaged hashes in the inventories. The current publication adds related-work findings identified during the final review.

## Model and runtime foundations

| Foundation | Credit and relationship |
|---|---|
| Model and speculative architecture | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and [DSpark](https://arxiv.org/html/2607.05147v1). The model and scheduling research are DeepSeek's work. |
| Quantized checkpoint | [mlx-community oQ2.5e-MTP revision `78456115`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab), using oMLX oQ. The downloaded weights are unchanged and are not redistributed here. |
| Array framework and model ecosystem | [Apple MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm) and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm). |
| Serving and native execution | [oMLX](https://github.com/jundot/omlx): model loading, native projection/storage paths, scheduler, speculative loop, caching and serving infrastructure. The measured starting environment was installed 0.7.1rc1. |

## Techniques, reused code and local additions

**BF16 vocabulary weight reuse.** The local kernel adapts oMLX's existing BF16 projection. Directly analogous multi-row reuse preserving the per-row reduction was already published in [tacos8me's head implementation](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py) before this exact-1017 work. Related 8-bit sharing appears in oMLX [#4475](https://github.com/jundot/omlx/pull/4475). Local additions include implementation details, integration, the explicit six-row extension and exact-checkpoint validation. The implementations are not byte-identical. Similarity establishes neither direct copying nor independent invention; no new weight-reuse algorithm is claimed.

**Confidence scheduling.** The causal stopping method follows [DSpark section 3.2.2](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2). An earlier [tacos8me confidence/cost policy](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/mtp.py) also uses offline costs and expected-token utility. The local implementation uses causal first-non-improvement stopping, previous-token conditioning, a zero-draft option and per-position calibration, with the measured costs frozen before evaluation. These are implementation and integration distinctions, not invention of confidence-based speculative scheduling.

**Dependency replay and numerical preservation.** Exact shrinking decoder suffixes and partition-preservation techniques appear in [antirez/ds4](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42408). Related MLX replay, hyper-connection row hints and draft capture appear in [tacos8me's encoder replay](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/encoder_replay.py). DeepSeek and related implementation work supply architecture/dependency analysis. The local contribution adapts replay to the pinned full-depth reference, handles projection/split-K, gate, hyper-connection and final expert-dispatch boundaries, preserves all three draft-stage projections, and integrates guards, fallback, state checks and lifecycle handling.

**HotRows.** The cache implementation is copied **unchanged** from [tacos8me's `hot_rows.py` at `fb233e2`](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/hot_rows.py), with its Apache-2.0 license and notice. The model-owned adapter, bounded budgets, reader coordination, lookup/gather integration, fallback, cleanup and associated measurements are local additions. No new row-cache algorithm is claimed.

**Engram prefetch and chunk configuration.** The selected-row reader and background executor are existing [oMLX storage infrastructure](https://github.com/jundot/omlx/blob/165be1ff5933a020f2f5a7544e1e95a2797f777f/omlx/patches/deepseek_v41/storage.py). The local work measures and integrates a larger admission budget with wider prompt chunks. Wider-chunk work already exists in [#3949](https://github.com/jundot/omlx/pull/3949), and [#4611](https://github.com/jundot/omlx/issues/4611) proposes overlapping storage/cache changes. These are configuration and integration findings, not a new prefetch algorithm.

**Attention-output rounding.** The optional compatibility patch implements reference behavior documented by DeepSeek and [gnahzhang's exact-1017 conversion/runtime work](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX/blob/c98bd97baca052f484c1d6ea6b6e8a8eced404ef/README.md). The local work investigates and implements that boundary in the selected runtime. It does not establish improved general answer quality.

**Reproduction and release engineering.** The source/evidence manifests, saved-data analyzers, activation checks, scoped integration, process/port ownership, loading headroom, cancellation/restoration records and public explanation were developed for this experiment. They do not imply those facilities were absent or defective in oMLX generally.

## Licenses and review provenance

Preserve the original per-file notices and supplied license texts. The reference DeepSeek port and BF16 head sources carry MIT notices; shared oMLX MTP code and the copied HotRows component carry their applicable Apache-2.0 notices. Source-derived changes retain the appropriate notices. Fixture licenses and any omitted artifacts are recorded with the package's source/evidence inventory.

AI coding agents assisted implementation, investigation, tests, evidence analysis and drafting. Separate local agents reviewed selected sources and saved results. Those checks are **local audits**, not external peer review or independent-machine replication. Maintainer review and external reproduction have not been substituted by the local audit process.

The release's technical value is the combined integration, concrete numerical and state repairs, exact-model evidence and reusable findings. Every source above remains part of that explanation; none of the individual prior techniques is presented as newly invented here.
