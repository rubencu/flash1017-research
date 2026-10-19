# Local research-repository preparation checks

The installable fork has a separate [runtime validation record](runtime-preview-01/),
including its exact wheel, default-suite results and real-serving smoke check.
The older preparation checks below apply to the retained historical archive.

All checks in [`preparation-01.json`](preparation-01.json) passed:

- The root archive verifier checked all 337 retained files and invoked the
  original package verifier. The archive matches the frozen preview byte for
  byte, including its original manifest.
- The historical source verifier checked 140 copied files and five generated
  source-kit files, Python syntax, recorded local dependency edges and frozen
  policy values. The five-file runtime patch reconstructed the archived
  candidate files byte for byte from the included historical baselines.
- The portable analyzer recalculated the saved comparison and matched the
  reports: 20.06083918786373% less geometric-mean first-token time, a failed
  fixed-512 useful-answer qualification and nine early-EOS observations.
- All 36 CPU-only analyzer contract tests passed.

Commands and individual logs are recorded in the receipt. Recalculated reports
are [`recalculated.txt`](recalculated.txt) and
[`recalculated.json`](recalculated.json). These checks used Python's standard
library and the system patch utility. They performed no model execution,
installation, service interaction or publication.

These are source/evidence integrity and saved-data checks. They do not establish
that the historical prototype launches on another machine, reproduce a GPU
measurement, qualify the current runtime fork or provide an external signature.
The immutable archive was not modified; new check output lives here.
