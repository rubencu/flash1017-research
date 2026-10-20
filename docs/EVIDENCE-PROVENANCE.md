# Evidence provenance

This reference maps the recorded protocols, analysis corrections, excluded
runs and source archives. For the measured build's results and methods, start
with the [article](ARTICLE.md), [performance study](RESULTS.md) or
[quality evaluation](QUALITY.md).

Raw responses, grades, protocols and executed sources retain their recorded
bytes and filenames. Editorial changes to guides do not change those records.
The evidence manifests distinguish authored documentation from exact copied
files; `COPY-PROVENANCE.json` maps each copy to its source hash.

## Build identities

| Record | Identity and scope |
|---|---|
| Measured runtime | Package `0.7.1rc1+flash1017.2`, source `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`; see [build identity](../validation/quality-v2-01/BUILD-IDENTITY.json). |
| Public comparison baseline | oMLX `v0.7.1rc1`, source `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`. This is not a measurement against a moving upstream `main`. |
| Installed package `0.7.1rc1+flash1017.1` | [Installation record](../validation/runtime-preview-01/README.md), [quality record](../validation/quality-v1-01/README.md), and [RUNTIME-LINK.json](../RUNTIME-LINK.json). The quality comparison failed with seven task regressions. |
| Prototype experiments | [Source and evidence archive](../archives/2026-10-18-preview/README.md), tied to the installed application's embedded runtime. Its performance figures do not apply to the measured wheel. |

Names such as `v2`, `baseline-02`, `candidate-01` and `draft-01` are retained
artifact identifiers. Their meaning comes from the bound protocols and hashes,
not their numbering. The archive is retained for audit, not as installation
instructions or evidence for the measured wheel's headline.

## Quality protocol and analysis

The [quality protocol](../validation/quality-v2-01/protocol.json) was frozen
before the measured fork generated its 618 responses. It requires exact text
and finish-reason parity, zero score regressions, no missing/error outcomes and
no candidate-only output-budget stops. It reuses a complete baseline response
file by hash, and both arms were freshly graded with the seeded evaluator.

The [baseline's original protocol](../validation/quality-v2-01/protocol-02.json)
required absolute zero truncation. That rule failed on the baseline itself,
which had four output-budget stops. The measured-build protocol explicitly
permits the same baseline stops while requiring exact parity and no additional
ones. This amendment preceded candidate generation. The report keeps the
absolute-rule failure in `original_acceptance` and the declared conservation
rule in `planned_acceptance`; a passing latter field does not erase the former.

The [initial paired analysis](../validation/quality-v2-01/pair-summary-final-01.json)
failed one integrity check: its reporter interpreted draft width as fixed MTP
depth, although the frozen protocol explicitly required adaptive MTP capped at
three. The [analysis correction](../validation/quality-v2-01/analysis-correction-02.json)
checks the declared mode, ceiling and saved runtime snapshots. The plan,
requests, generated responses, grades and acceptance rules were unchanged;
no inference or grading was repeated. Both reporter sources and reports remain
in the bundle. The accepted
[paired report](../validation/quality-v2-01/pair-summary-final-02.json) binds the
exact reporter under `analysis-source-02/`.

The [evaluator specification](../validation/quality-v2-01/evalplus/AMENDMENT.md)
records the worker flags and seeds. The original evaluator's Python `-I`
workers ignored `PYTHONHASHSEED`; the seeded evaluator uses `-s -P -B` and an
allowlisted environment. Fresh validation and both sets of model grades use
that same seeded source. The original evaluator and scores are retained in
[their bundle](../validation/quality-v1-01/README.md). Neither execution setup
controls every source of nondeterminism. The two canonical-test exceptions and
540/542 canonical score remain disclosed in the [quality guide](QUALITY.md).

## Performance execution records

The [qualified comparison](../validation/performance-v2-01/qualified/pair-summary-final-01.json)
contains all four planned processes and 32 measured requests. It uses no
observations from the excluded attempt described below.

An [excluded attempt](../validation/performance-v2-01/interrupted/state.json)
stopped after two processes and 16 measured requests when another model began
loading in the regular server during the memory-headroom check. Its
[report](../validation/performance-v2-01/interrupted/pair-summary-final-01.json)
remains failed with null aggregates. The complete comparison repeated all four
processes with the same plan and supervisor source, rather than selecting
successful or favorable cases from the partial run. The qualified supervisor
record confirms all child processes exited and the regular server's model was
restored.

The [reproduction guide](../validation/performance-v2-01/README.md#reproduce-from-saved-files)
explains the full audit. Add `--attempt interrupted` to that command to audit
the excluded records. Expected exit status is 1, with missing second-block
files and no aggregate; temporary staging changes error paths. This verifies
the failed record without turning it into a passing result.

The preparatory fixture filenames retain `draft-01`, but their exact hashes
were frozen in the final protocol before any measured run. The HTTP inputs
were rendered with the pinned chat encoder; compared with archived direct-run
inputs, CPython 64K has different token IDs and the novel excerpt is 16,383
tokens. The supplied HTTP requests and tokens define this study.

## Source archive and additional techniques

The [archive index](../ARCHIVE-INDEX.json) binds 337 files, including prototype
experiments, unsuccessful checks, component studies and source snapshots.
The [archive guide](../archives/2026-10-18-preview/DEPENDENCIES.md) describes
its environment requirements. Its 20.06% first-token figure, 7.34% head study
and 40.33% replay/cache result use different code or references and are not
combined with the measured wheel's 7.9% generation and 4.5% first-text results.

That archive includes investigations of numerical rounding, confidence
scheduling, decoder replay, shape-sensitive dispatch and draft-state priming.
Those paths are not active in the measured profile. Related implementations
and their exact credits are mapped in the
[archived attribution record](../archives/2026-10-18-preview/docs/ATTRIBUTION.md).
The [recorded preparation checks](../validation/preparation-01.json) and
[recalculation](../validation/recalculated.json) concern the archive only.

Private server settings, model weights and Python environments are excluded
from the distributed evidence. Saved file verification and offline arithmetic
are not independent inference replication or external attestation.
