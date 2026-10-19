# Repository layout and artifact identities

The project has two repositories so that users can install and develop the
runtime while researchers can inspect the exact code and observations behind
each measurement.

| Repository | Contents and use |
|---|---|
| `rubencu/omlx` | The maintained oMLX fork: runtime source, installation instructions, configuration, tests, release builds and branches for focused upstream changes. Runtime contributions belong here. |
| `rubencu/flash1017-research` | Methods, articles, raw observations, analysis tools, figures, attribution and source snapshots tied to particular experiments. Analysis corrections and additional studies belong here. |

**Availability:** the oMLX fork exists, but the `flash1017-preview` branch and
wheel are prepared locally and have not been pushed. The research remote is
proposed and has not been created. In the local review workspace, the runtime
checkout is the sibling `../omlx`.

## Which code produced which result?

The **benchmark prototype** ran in the installed oMLX 0.7.1rc1 application's
embedded environment. Its combined latency comparison and component studies
are described in [Methods and results](RESULTS.md). The corresponding source,
input tokens, saved observations and reports are under
`archives/2026-10-18-preview/`.

The **installable fork** integrates the retained prototype changes into normal
package modules. Its branch is `flash1017-preview`, based on public oMLX
`v0.7.1rc1` at `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`. Package version
`0.7.1rc1+flash1017.1` passed clean installation and a bounded serving check.
[`RUNTIME-LINK.json`](../RUNTIME-LINK.json) records its exact source commit,
wheel checksum, checkpoint revision and [validation directory](../validation/runtime-preview-01/README.md).

The public native binaries packaged with the fork differ from the installed
application's binaries used in the benchmark. The prototype's performance
measurements therefore describe that prototype, not the installable wheel.
This distinction lets readers identify both the software they can run and the
software actually measured.

## How to use the research archive

The archive contains source snapshots, adapters used to launch and measure the
prototype, recorded calibration, a five-file patch against the public release
baseline, and evidence from the overall and component experiments. Its
[source/dependency guide](../archives/2026-10-18-preview/DEPENDENCIES.md)
explains what is included and what inference reproduction requires.

Development notes inside the archive retain their original filenames, internal
profile names and release proposals. Use the [article](ARTICLE.md) and
[methods document](RESULTS.md) to interpret them. The archive's launch scripts
still reflect their application-dependent environment; they are evidence of
the experiment rather than the fork's installation instructions.

`ARCHIVE-INDEX.json` binds the 337 retained files, including the archive's own
manifest. To preserve the link between measurements and source, keep that
snapshot unchanged. Add new studies as new versioned records identifying their
source, build, hardware, method and raw observations. Update the maintained
runtime in the oMLX fork, rather than editing a measured source snapshot here.

## Releases and upstream contributions

Installation instructions and release downloads belong in `rubencu/omlx`.
Research articles should link to the exact evidence snapshot and identify the
build covered by any performance claim. A new Hugging Face model repository
is unnecessary because this work leaves the checkpoint weights unchanged.

Focused upstream branches originate from the runtime fork and need validation
against the target upstream revision. The archive's `upstream/bf16-rows2-5/`
folder is a draft patch prepared against a different upstream revision; it is
neither a submitted PR nor a benchmark of current upstream `main`.

The [publication plan](PUBLICATION-PLAN.md) specifies the proposed releases and
upstream work. Remote creation, pushes, releases and submissions await the
repository owner's review.
