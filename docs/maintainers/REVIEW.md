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

The research remote is not yet created and the runtime preview has not been
pushed. Verify availability statements and links when publication is approved.
The runtime and research repositories should each point to the other's exact
release or evidence record.
