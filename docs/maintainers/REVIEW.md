# Release review checklist

This document is for the repository owner. Publication actions require the
owner's approval; the checklist itself does not authorize them.

The measured runtime is package `0.7.1rc1+flash1017.2`, source commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. Its exact source and wheel hashes
are recorded in the [build identity](../../validation/quality-v2-01/BUILD-IDENTITY.json).

Before publication, review:

1. The [article](../ARTICLE.md), [performance methods](../RESULTS.md) and
   [quality evaluation](../QUALITY.md), including the claims' hardware and
   workload scope.
2. The quality and performance bundles, raw data, reproduction commands and
   [evidence provenance](../EVIDENCE-PROVENANCE.md).
3. [Attribution](../ATTRIBUTION.md), per-file licenses and corpus notices.
4. The runtime installation instructions and release wheel hash; documentation
   changes must not be mistaken for a newly qualified runtime build.
5. The exact repository, branch, tag, release assets, article destination and
   any upstream submissions proposed in the [publication plan](PUBLICATION-PLAN.md).

## Prepared publication

| Repository | Branch | Experimental release tag |
|---|---|---|
| Existing `rubencu/omlx` fork | `flash1017-preview` | `v0.7.1rc1-flash1017.2` |
| New public `rubencu/flash1017-research` | `main` | `v0.1.0` |

Review the exact [runtime release text](https://github.com/rubencu/omlx/blob/v0.7.1rc1-flash1017.2/docs/releases/flash1017-2.md),
[research release text](../releases/v0.1.0.md) and [article](../ARTICLE.md).
The runtime release carries the preserved wheel, version constraints and
checksums. The research release carries the complete source/evidence archive.
The [publication plan](PUBLICATION-PLAN.md) lists the operations and assets.

The [installation record](../../validation/release-install-01/README.md) confirms
a clean environment with all 110 measured dependency versions, all four pinned
Git revisions and all 752 runtime/web files matching the tested wheel. The
saved quality/performance evidence and offline full audit also pass. These
packaging checks do not add a new performance or quality claim.

The research remote is not yet created and the runtime preview has not been
pushed. Prepared reader-facing links use the exact planned release tags and
become live upon publication. No Hugging Face blog, social post, upstream PR,
fork default-branch change or replacement app/Homebrew release is included.
