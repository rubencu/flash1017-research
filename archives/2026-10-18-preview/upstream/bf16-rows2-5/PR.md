# Draft PR: reuse BF16 vocabulary weights across verification rows

**Local draft only. Hardware validation is pending; no PR has been submitted.**

DeepSeek V4.1's specialized BF16 vocabulary projection reads the same weight row separately for each verification query. This change shares that read across two to five queries, keeping a separate accumulator for each and preserving the intended per-query operation order.

The existing eligibility guard is unchanged. One-row calls retain the original kernel; CPU, other dtypes, unsupported dimensions and more than five rows retain their original fallback. Six-row arithmetic, attention rounding, confidence scheduling and other Flash1017 features are outside this change.

This adapts the kernel used in the Flash1017 experiment into the production projection function. The Metal source body is unchanged from that local implementation. A directly analogous BF16 multi-row implementation appeared earlier in [tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py); related 8-bit weight sharing landed in [#4475](https://github.com/jundot/omlx/pull/4475). No algorithmic novelty is claimed.

## Validation status

Base: `165be1ff5933a020f2f5a7544e1e95a2797f777f`.

Completed CPU-only packet checks: pinned-file integrity, clean patch application, Python syntax compilation without module execution, unchanged guard/import/original-kernel structure, and identity of the reused local Metal source.

**Not run:** Metal compilation, actual `project_logits` execution, current-main numerical comparison, the upstream focused/default test suites, or paired current-main real-model performance checks. The [validation checklist](VALIDATION.md) specifies these remaining gates. This draft is not ready to merge.

The earlier controlled local head study observed approximately 7.2–7.5% higher generation throughput with matching output tokens, but both arms used local reference rounding and fixed depth two. It is historical supporting evidence, **not this patch's current-main benchmark**. No speedup is claimed for the extracted patch until it is measured.
