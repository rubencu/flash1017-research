# Evidence checks and runtime validation

This repository separates evidence integrity, model quality, and functional
software checks. Passing one does not establish the others.

| Record | Question answered |
|---|---|
| [Completed installed v2 performance](performance-v2-01/README.md) | How does the same wheel perform over eight HTTP workloads? **7.9% higher generation throughput and 4.5% lower first-text latency, with all 16 output pairs matched.** |
| [Completed installed v2 quality evaluation](quality-v2-01/README.md) | Does the revised wheel preserve the tested baseline responses? **Yes: all 618 texts and finish reasons match, with zero task regressions under the frozen amended rule.** |
| [Completed installed v1 quality evaluation](quality-v1-01/README.md) | Does the complete v1 wheel preserve the baseline's task outcomes? **No: seven observed paired regressions; the frozen gate failed.** |
| [Historical v1 runtime validation](runtime-preview-01/README.md) | Does the identified wheel install, pass its recorded tests and serve requests on the test machine? |
| [Research preparation receipt](preparation-01.json) | Do the archived files match their manifests, and does the analysis reproduce the reported numbers from the saved observations? |

## Completed v2 quality evidence

The local `0.7.1rc1+flash1017.2` build at
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1` passed its bounded quality plan:
618 exact answer-text and finish-reason pairs, 618 paired grades and zero
baseline-pass to candidate-fail tasks. Both arms passed 148/164 HumanEval+,
299/378 MBPP+, 61/64 selected GSM8K questions and 11/12 synthetic long-context
controls. Four identical baseline output-budget stops remain included.

The protocol reuses the exact original baseline response file by hash;
baseline inference was not repeated. Both arms' coding answers were freshly
graded with the same amended seed-controlled worker. Canonical validation
remains 540/542 with two pinned upstream exceptions retained in the denominators.
The first v2 report failed a reporter check that misread adaptive MTP as fixed
depth. The analysis-only correction and both reports remain inspectable;
requests, outputs, grades and frozen acceptance rules did not change.

The [v2 bundle guide](quality-v2-01/README.md) maps all evidence, exact source
versions and provenance limits. Its corrected report records 15,900 passed
integrity checks. Verify its saved bytes with:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v2-01
```

This verifies the bundle, not universal model equivalence or an independent
inference rerun. The original zero-truncation rule remains failed. Historical
[`RUNTIME-LINK.json`](../RUNTIME-LINK.json) is unchanged; the current identity
is [quality-v2-01/BUILD-IDENTITY.json](quality-v2-01/BUILD-IDENTITY.json).

## Completed v2 performance evidence

The full HTTP comparison passed 16,193 integrity checks: 32 measured requests,
four excluded warmups and 16 matched workload/block pairs. Across eight mixed
workloads, generation throughput was 7.9% higher and first-visible-text latency
4.5% lower. The six natural-source cases had 8.0% higher throughput and 5.5%
lower first-text latency. Generation improved in all 16 pairs; first-text
latency improved in 11/16 with noticeable block variation.

Only three workloads ended normally within 512 tokens, giving six pairs for
completed-answer latency. The other requests remain in first-text and
throughput results. These are two-block measurements on one Mac, without
confidence intervals or general hardware claims. A prior attempt interrupted
after 16 requests remains failed with no aggregate; the successful comparison
repeated all four processes under the unchanged plan. Original model restoration
completed.

The [performance guide](performance-v2-01/README.md) provides all raw results,
frozen sources and two CPU-only reproduction commands: raw-measurement arithmetic
without binaries, and the complete frozen audit with the two exact wheels.
Neither requires model weights or private settings. The old prototype's 20.06%
result is a separate historical comparison.

## Historical v1 quality evidence

All 618 requests and paired grades are retained in `quality-v1-01`. Coding
passes were 447/542 in each arm, with five regressions and five gains; GSM8K
subset passes were 61/64 in each arm; synthetic long-context controls changed
from 11/12 to 10/12, with two regressions and one gain. The seven regressions
violate the original zero-regression rule. Four baseline and three candidate
length stops also violate its absolute zero-truncation rule.

Canonical coding validation passed 540/542, with two documented pinned
upstream exceptions retained in both denominators. Original isolated-worker
hash randomization limits deterministic regrading. A later seed-controlled
protocol remains separate from the original source, scores and failed gate.
The [bundle guide](quality-v1-01/README.md) explains these limits and the exact
file inventory. Verify the saved bytes with:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v1-01
```

The [quality document](../docs/QUALITY.md) explains the revised runtime and the
separate v2 rule. Neither the amended grading protocol nor the v2 result
overwrites the original v1 outputs, scores or failed gate.

## Research archive and analysis checks

All checks in `preparation-01.json` passed:

- The archive verifier checked all 337 retained files and invoked the archive's
  internal package verifier. The file contents match the frozen snapshot.
- The source verifier checked 140 copied files and five generated source-kit
  files, Python syntax, dependency references and fixed policy values. The
  five-file runtime patch reconstructed the archived candidate files byte for
  byte from the included baseline files.
- The analyzer recalculated the overall comparison: **20.06083918786373% less
  geometric-mean time to first token**, nine early end-of-sequence observations
  and failure of the 512-token answer-completion acceptance rule. The
  [methods document](../docs/RESULTS.md) explains that rule and its consequence.
- All 36 CPU-only analyzer contract tests passed.

Commands and individual logs are recorded in the receipt. Recalculated reports
are [`recalculated.txt`](recalculated.txt) and
[`recalculated.json`](recalculated.json). These checks use Python's standard
library and the system patch utility. They require no model execution,
installation or server interaction.

File verification and saved-data analysis test the integrity and interpretation
of the recorded experiment. Repeating GPU inference, measuring the installable
fork and reproducing the results on another machine require additional work.
The receipt is a local check, not an external signature or independent
replication.
