# Two repositories, distinct responsibilities

| Repository | Responsibility |
|---|---|
| `rubencu/omlx` | Canonical maintained oMLX fork: runtime source, installation, user-facing configuration, runtime tests, releases and focused upstream PR branches. Local sibling: `../omlx`. |
| `rubencu/flash1017-research` | Research record: articles, methods, raw observations, analysis tools, figures, attribution and immutable source snapshots tied to measured experiments. This remote is proposed and has not been created. |

The runtime fork contains the installable distribution for users
and continuing development. The research repository is not an installer or a
second implementation to edit in parallel. Change runtime code in the fork;
archive the measured source identity here when a study or release needs it.

## Historical snapshot

`archives/2026-10-18-preview/` is an exact copy of the frozen reviewed artifact.
Its source includes all accepted local prototype changes, external AST-loaded
definitions, calibration, serving and benchmark adapters, and the five-file
patch against the public historical oMLX baseline. Its installed-app dependency
and source guards are preserved. It is not the current fork's source checkout.

Its manifests continue to describe its original files without being copied to
the root as a stale live manifest. `ARCHIVE-INDEX.json` binds all retained bytes,
including the historical manifest itself. Keep this directory unchanged. New
results should get a new versioned archive rather than silently replacing old
measurements, failed qualifications or source files.

## Runtime linkage

The initial local runtime work uses branch `flash1017-preview`, based on public
oMLX `v0.7.1rc1` commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`.
This preserves the historical API/build baseline and provides a public native
wheel route for independent installation checks. The integrated fork has passed
a clean installation and bounded real-serving check; see the recorded scope.
This branch/base choice is distinct from the archived narrow
BF16 upstream packet, which was prepared against a later upstream revision.

`RUNTIME-LINK.json` records the exact runtime commit and wheel identity, with
separate evidence under `validation/runtime-preview-01/`. The historical
prototype did not measure this newly packaged fork. A source port or successful installation alone does not
transfer the old 20.06% or 40.3% performance measurements to a different artifact.

Maintain two separate facts:

- Which fork commit a user should install and what checks it passed.
- Which exact source/build each historical or future measurement describes.

The first may change over time; immutable historical evidence should not.
Do not label old measurements as current-fork benchmarks until the actual
shipped artifact has been measured under an explicitly recorded method.

## Publication and upstream work

Prepare both repositories locally, with reviewable code, docs, license/credit
records and validation. Human review precedes remote creation, pushes, releases,
articles or upstream discussions/PRs. Runtime installation instructions and
release downloads belong in `rubencu/omlx`; research articles should link to
that runtime and to a specific evidence snapshot here.

Focused upstream submissions originate from the runtime fork. The archived
`upstream/bf16-rows2-5/` packet is historical preparation, not an already submitted
PR or a current-main runtime-performance validation. Avoid copying the entire research archive into an
upstream diff. No new Hugging Face model repository is needed because this work
does not change the existing checkpoint's weights or quantization.
