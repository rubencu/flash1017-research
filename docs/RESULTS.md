# What the historical findings establish

These measurements describe the **October 18 prototype**, not the current
runtime fork. The source, data and original reports are retained unchanged in
[`archives/2026-10-18-preview`](../archives/2026-10-18-preview/README.md).

## Complete prototype versus the installed application

The exact checkpoint was `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`,
revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. Hardware was one M5 Ultra
Mac Studio with 80 GPU cores and 256 GiB unified memory. The reference was the
original installed oMLX 0.7.1rc1 with native adaptive MTP enabled, not current
upstream main or literal MTP-off factory defaults.

Across eight fixed workload definitions, three configurations and two process
blocks, all 48 measured requests completed. Six declared warmups were excluded.
Equal-case geometric-mean first-token time fell **19.554 → 15.631 seconds,
20.06% less** against stock 2K chunks. Prompt-processing time fell 20.17%.
The 8K-chunk stock control was slower; its larger relative improvement is not
the primary headline. The combined result includes configuration and code and
does not isolate individual patch contributions.

These were warm-runtime measurements with fresh conversation state. Models
stayed loaded, OS caches were not purged, and candidate row caches persisted
within each measured block after their declared reset. Timing excludes model
loading and HTTP transport. The timing suite's longest input was 64K tokens.

Preserved limitations:

- The 64K CPython first-token case regressed about 2.32%; the short 1K aggregate
  improved only about 0.60%. A fixed-suite average is not every user's speedup.
- Sampled peak process footprint increased **208.35 → 216.10 GiB**, about
  7.75 GiB. The changed chunk geometry is part of that difference; it cannot all
  be attributed to the 512 MiB row cache. Sampling may miss brief peaks.
- The original fixed-512 useful-answer qualification **failed**: nine requests
  reached EOS early, three in each arm. Forced post-EOS work is not valid useful
  answer-completion time. These are not nine candidate crashes or nine proven
  incorrect answers. All observations are retained.
- First-token and prefill timing qualify separately. The **20.21% less time to
  token 128** supplement is **post hoc**, using delivery timestamps within the
  original executions. It is not a new 128-token benchmark or completion time.
- Outputs differ between the complete and stock configurations. General quality,
  output-distribution equivalence and faster successful-task completion are
  unestablished. Two blocks on one machine are not external replication, a
  confidence interval or evidence for other hardware and workloads.

See the [complete report](../archives/2026-10-18-preview/evidence/overall/review/report.txt)
and [analysis contract](../archives/2026-10-18-preview/analysis/README.txt).

## Earlier studies remain separate

The 40.3% lower prefill-time result was a 12-pair replay-plus-cache comparison
against an already corrected local `native64` reference, with 1,536 matching
output positions. That reference already included earlier local changes.
It is not an overall result versus untouched oMLX and is not additive to 20.06%.
Component figures use their own denominators and must not be multiplied together.

The historical optimized serving profile had a 131072 total-context cap and
one active request with queuing. Separate offline state/output checks at longer
lengths do not establish optimized million-token chat. The long-HTTP gate
failed after a recovery response inserted an unexpected newline; its cause was
not established and the remaining long cases were not run. Failed, negative and
undeployed studies remain in the [historical record](../archives/2026-10-18-preview/evidence/historical/README.md).

## What has not yet been established for the current fork

The archived package is an auditable prototype snapshot with portable saved-data
analysis. Its original launch depended on an installed application, native ABI,
lab-relative helpers and historical hash guards. The public baseline is partly
recoverable, but a public wheel was not byte-identical to the measured native
modules; this does not itself prove incompatibility.

The companion fork's source/build identity, clean installation, default-suite
results and bounded real-serving check are recorded separately in
[runtime-preview-01](../validation/runtime-preview-01/). Those checks establish
a functional preview. This repository makes **no current-fork performance or
answer-quality equivalence claim**, nor a first-on-Mac, fastest-engine,
all-model/all-Mac or new-quantization claim.
