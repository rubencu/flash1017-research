# Flash-1017 on a 256 GiB Mac: 20.1% less time to first token

**A research preview of an oMLX configuration for DeepSeek-V4.1-Flash-1017.** In an eight-case warm-runtime suite on one M5 Ultra, the complete configuration reduced geometric-mean time to first token from **19.554 to 15.631 seconds** against installed **oMLX 0.7.1rc1 with native speculative decoding enabled**. This is a measured local implementation, with source and evidence to inspect. A standalone installer has not been verified.

The gain varies by workload. One 64K code case regressed, sampled peak memory increased by **7.75 GiB**, and this experiment does **not** establish equivalent answer quality. The comparison is against the installed version, not current upstream `main`.

![First-token latency by workload and configuration](../evidence/overall/review/figures/first-token.png)

*Two fresh-process blocks per configuration; 48 measured requests in total. The figure's block ranges are not confidence intervals. Timings exclude model loading and HTTP transport.*

## What changed

The model already ran in ordinary oMLX. We used the existing [mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab), unchanged. The work concerns how that checkpoint executes:

- Reuse BF16 vocabulary weights across several verification rows.
- Apply the documented attention-output reference rounding in both target and draft paths.
- Use DSpark confidence to select a verification prefix, with a frozen cost and calibration policy.
- Use 8,192-token prompt chunks and a 64 MiB admission budget for oMLX's existing background Engram reader.
- Reduce decoder work using dependency-aware replay, preserving the relevant numerical behavior and speculative draft state.
- Integrate a bounded HotRows cache with model ownership, native-reader coordination, fallback and cleanup.

These changes include adaptation of existing community implementations and published techniques. The contribution is the exact-checkpoint integration, concrete numerical diagnosis and repairs, and the measurements showing what did and did not work. The [attribution map](ATTRIBUTION.md) distinguishes copied code, related implementations and local additions.

## Why removing rows was not enough

Some decoder work does not contribute to the final outputs we need. Dependency-aware replay computes progressively smaller suffixes instead. That idea already appears in [ds4](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42408) and related MLX work.

The difficult part was preserving the intended floating-point computation. The number of rows can determine which GPU kernel runs. Removing mathematically unnecessary rows can therefore change rounding in the rows that remain.

One concrete example appeared at the final mixture-of-experts layer. A replay suffix had **255 rows**. With six selected experts per row and 384 experts, that produced 1,530 routes—just below a grouped-kernel dispatch threshold of 1,536. Adding one temporary zero row crossed the threshold and restored the relevant dispatch behavior. The implementation discards that extra output row; it does not add a token to attention or the cache. This was a targeted repair for a known model layout, not a general instruction to pad every calculation.

We also found that matching the target model's state was insufficient. DSpark, the speculative drafter, needs its own prompt context. The target state could match while the physical draft rings differed. The final implementation preserves the relevant arithmetic in all three draft-stage projections, crops before the subsequent normalization and cache append, and checks that priming completes. That is why the tests inspect both target and draft state, as well as generated tokens.

These checks support the tested execution paths. They are not a universal equivalence proof for every model, shape or request.

## The complete comparison

All three configurations used the same downloaded weights, hardware and embedded dependency environment. Both stock configurations used native adaptive MTP with a maximum depth of three. The complete configuration included all changes listed above, including its five-token draft width and numerical policy.

| Configuration | Prompt chunks | First-token time, geometric mean | Sampled peak process footprint |
|---|---:|---:|---:|
| Installed oMLX, native MTP | 2,048 | 19.554 s | 208.35 GiB |
| Same installed runtime and MTP | 8,192 | 24.293 s | 215.77 GiB |
| Complete local configuration | 8,192 | **15.631 s** | **216.10 GiB** |

The 2K stock arm is the stronger aggregate reference. Widening chunks alone was slower here; the larger percentage against stock 8K is not the headline.

The suite contains two synthetic 1K code/prose cases, four natural-source 16K cases, and two 64K cases, using CPython, curl, *Moby-Dick* and *Pride and Prejudice*. Exact input tokens were frozen. Each arm ran in two fresh processes; the second block reversed configuration and case order. Six warmups were excluded. Models remained loaded and OS caches were not purged during measurement. Each request had fresh conversation state, and the candidate's row cache persisted through the declared measured sequence.

At 16K, aggregate first-token time fell **32.19%**. At 1K, it changed only **0.60%**. The 64K CPython case became **2.32% slower** to first token, from 102.073 to 104.445 seconds. The complete configuration also used more memory. The stock 8K arm was already close to its footprint, so the full 7.75 GiB difference from stock 2K cannot be assigned to the 512 MiB retained-row cache.

## A qualification that failed

The benchmark requested 512 generated tokens with stopping disabled. **Nine requests reached EOS before token 512: three in each arm.** That failed the original fixed-512 useful-answer qualification. These are not nine candidate crashes or nine demonstrated wrong answers, but computation after EOS cannot establish useful answer-completion speed.

All observations remain in the archive. Prefill and first-token measurements qualify separately because they precede EOS. A supplementary analysis found **20.21% less time to token 128**, with every observed prefix preceding EOS. That analysis is **post hoc** and uses delivery timestamps inside the original runs; it is not a new 128-token request benchmark. Forced-512 rates remain descriptive and are not the main performance claim.

The [full report](../evidence/overall/review/report.txt) includes the protocol, per-case results, resource observations and limitations. Different output streams also mean the complete comparison cannot establish unchanged answer quality or faster equivalent task completion.

## What the component studies tell us

An earlier controlled head-kernel study observed approximately **7.2–7.5% higher generation throughput**, with all 10,240 output-token positions matching. Both sides already used the local reference-rounding configuration. It supports a focused BF16 row-reuse contribution; it is not a current-main PR benchmark. Rows two through five preserve the old shape-matched arithmetic. Extending the kernel to six rows replaces a different upstream fallback and must be treated separately.

A later replay-plus-cache study observed **40.3% less prompt-processing time**, with 1,536 matching output positions. Its reference already included the earlier head, confidence, rounding and prefetch work. That is an incremental result, distinct from the complete comparison above. The component percentages cannot be added or multiplied.

## What is being shared

This preview provides the [accepted implementation and supporting source](../source/), [saved-data analysis](../analysis/), and the evidence needed to inspect the results. Recalculating saved results is different from rerunning the model. The historical application-dependent launch is not a verified standalone installer.

The local serving profile is limited to the pinned checkpoint, one active request, this tested 256 GiB M5 Ultra, and a **131,072-token total context limit**. The overall timing suite stops at 64K input. Offline longer-context checks do not qualify optimized million-token chat; the separate failed long-serving gate remains in the record.

The immediate upstream opportunity is a small, credited BF16 row-reuse adaptation with production-path tests. Reference rounding is a separate conditional compatibility change. Confidence, replay and storage integration need more substantial scoped integration and coordination. The [upstream roadmap](UPSTREAM.md) states what evidence exists and what is still missing.

The useful result is an inspectable improvement for a specific configuration, together with an explanation of the numerical and state boundaries that made it difficult to implement. Other developers can reproduce the arithmetic, challenge the measurements, and adapt the pieces that fit their systems.
