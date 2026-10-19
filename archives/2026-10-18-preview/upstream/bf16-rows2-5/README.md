# BF16 head row reuse: local upstream patch packet

**Reviewable proposal. Not submitted; not merge-ready. No current-main GPU or real-model validation has been run for this patch.**

[change.patch](change.patch) modifies one file in oMLX at commit [`165be1ff5933a020f2f5a7544e1e95a2797f777f`](https://github.com/jundot/omlx/tree/165be1ff5933a020f2f5a7544e1e95a2797f777f). It introduces weight reuse only for **two through five BF16 query rows admitted by the existing guard**. The original one-row kernel, imports, guard and fallback return are retained. Six rows still take the original fallback. No confidence, rounding, replay, cache, model-setting or scheduler changes are included.

The Metal kernel body is reused byte-for-byte from the [packaged local research kernel](../../source/outputs/flash1017-lab/flash1017/head.py), with the Python constant renamed `_ROWS_SOURCE`. This is a production-function adaptation of the existing experiment, not a new kernel variant. The [pinned base file](base/omlx/patches/deepseek_v41/head.py) retains its MIT notice; the package also supplies the [applicable reference-source license text](../../source/outputs/flash1017-lab/third_party/deepseek-v41-MIT.txt).

A directly analogous [BF16 row-reuse implementation](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py) predates this work and deserves credit. The local kernel has different load/reduction-loop details. Similarity alone establishes neither direct copying nor independent development. Related merged oMLX [#4475](https://github.com/jundot/omlx/pull/4475) concerns 8-bit projections.

## Contents

- [BASE.json](BASE.json): pinned source URL, base/patch/proposed-file hashes and kernel provenance.
- [change.patch](change.patch): minimal production-file diff.
- [PR.md](PR.md): draft description with honest validation status.
- [VALIDATION.md](VALIDATION.md): finite missing validation requirements.
- [verify.py](verify.py) and [verification.json](verification.json): CPU-only application, syntax and source-integrity checks.

From the release package root:

```sh
python3 upstream/bf16-rows2-5/verify.py
```

This uses ordinary Python and Git. It applies the patch to the pinned file in a temporary directory, compiles Python syntax without executing the module, and checks source/AST preservation. It neither imports MLX nor compiles Metal, invokes a production model function or runs inference. These are packet checks, not numerical or performance tests.

Historical head measurements in the research release are supporting context only. Their reference already contained local attention rounding and their fixed-depth-two execution does not constitute a current-main benchmark across all widths two through five. None of the overall package's 20.1% or historical replay/cache 40.3% results is attributed to this patch.
