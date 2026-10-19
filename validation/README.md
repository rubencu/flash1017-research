# Evidence checks and runtime validation

This repository records two kinds of verification:

| Record | Question answered |
|---|---|
| [Installable runtime validation](runtime-preview-01/README.md) | Does the identified wheel install, pass its recorded tests and serve requests on the test machine? |
| [Research preparation receipt](preparation-01.json) | Do the archived files match their manifests, and does the analysis reproduce the reported numbers from the saved observations? |

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
