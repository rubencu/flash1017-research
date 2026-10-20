# Inspect and reproduce the results

The evidence covers package `0.7.1rc1+flash1017.2` and public oMLX `v0.7.1rc1`
running the same DeepSeek-V4.1-Flash-1017 checkpoint on a 256 GiB M5 Ultra Mac.

| Question | Evidence |
|---|---|
| How much faster is the fork? | [Performance](performance-v2-01/README.md): 7.9% higher generation throughput and 4.5% lower first-visible-text latency across eight workloads, 32 measured requests and two blocks. |
| Did the tested responses change? | [Answer quality](quality-v2-01/README.md): all 618 response texts and finish reasons match, with zero task regressions. |
| Which exact software was measured? | [Build identity](quality-v2-01/BUILD-IDENTITY.json): source, wheels, checkpoint and profile. |
| Does the release install with the measured dependencies? | [Fresh installation](release-install-01/README.md): 110 dependency versions and 752 runtime/web files checked against the measured environment and wheel. |

## Check the saved evidence

From the research repository, using Python 3.9 or newer:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v2-01
python3 -I -B tools/verify_quality_evidence.py --folder validation/performance-v2-01
python3 -I -B validation/performance-v2-01/reproduce.py --mode measurements --output /tmp/flash1017-performance-measurements.json
```

The first two commands verify file inventories and checksums. The third
recomputes the timing ratios and aggregates from saved responses and checks
paired output equality. These commands require neither model weights nor a
running server, and do not execute generated answers.

For the full performance audit, including the saved runtime identities,
follow the [wheel-based reproduction instructions](performance-v2-01/README.md#reproduce-from-saved-files).
The [quality guide](quality-v2-01/README.md) explains task sources, scoring and
the resources needed to repeat generation or execute the coding tests.

## Interpret the checks

File verification establishes that the evidence has not changed. Offline
analysis reproduces calculations from that evidence. Repeating model inference
on another machine is a separate experiment.

The results cover one machine and fixed workloads. Four quality responses
reached their output budgets identically in both builds. Five timing workloads
reached the 512-token budget, so complete-answer latency applies only to the
other three. The bundle guides state these limits alongside the results.

The [provenance record](../docs/EVIDENCE-PROVENANCE.md) maps analysis corrections,
excluded attempts and source archives for readers auditing the evidence.
