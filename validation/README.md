# Evidence checks and runtime validation

This repository separates evidence integrity, model quality, and functional
software checks. Passing one does not establish the others.

| Record | Question answered |
|---|---|
| [Completed installed v1 quality evaluation](quality-v1-01/README.md) | Does the complete v1 wheel preserve the baseline's task outcomes? **No: seven observed paired regressions; the frozen gate failed.** |
| [Historical v1 runtime validation](runtime-preview-01/README.md) | Does the identified wheel install, pass its recorded tests and serve requests on the test machine? |
| [Research preparation receipt](preparation-01.json) | Do the archived files match their manifests, and does the analysis reproduce the reported numbers from the saved observations? |

## Completed v1 quality evidence

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

The current local `0.7.1rc1+flash1017.2` candidate at
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1` has a revised stock-arithmetic
profile. Its complete qualification is pending; it has no measured performance
result. [`RUNTIME-LINK.json`](../RUNTIME-LINK.json) remains the historical v1
identity. The [quality document](../docs/QUALITY.md) states the current gates.

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
