# Release review and publication

The owner reviewed and approved the GitHub publication. Both prereleases were
published on 2026-10-19. The [publication receipt](PUBLISHED.json) records exact
release identities and verified public-download hashes.

| Repository | Branch | Published experimental release |
|---|---|---|
| `rubencu/omlx` | `flash1017-preview` | [v0.7.1rc1-flash1017.2](https://github.com/rubencu/omlx/releases/tag/v0.7.1rc1-flash1017.2) |
| `rubencu/flash1017-research` | `main` | [v0.1.0](https://github.com/rubencu/flash1017-research/releases/tag/v0.1.0) |

The reviewed publication consists of the [article](../ARTICLE.md),
[performance methods](../RESULTS.md), [quality evaluation](../QUALITY.md),
[runtime release notes](https://github.com/rubencu/omlx/blob/v0.7.1rc1-flash1017.2/docs/releases/flash1017-2.md)
and [research release notes](../releases/v0.1.0.md). The
[distribution record](PUBLICATION-PLAN.md) lists released assets and identifies
upstream work that remains separate.

The measured package is `0.7.1rc1+flash1017.2`, runtime source
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. Its exact source and wheel hashes
are recorded in the [build identity](../../validation/quality-v2-01/BUILD-IDENTITY.json).
The release tag adds documentation, dependency constraints and workflow guards
while preserving the tested runtime files and wheel.

The [installation record](../../validation/release-install-01/README.md) confirms
a clean environment with all 110 measured dependency versions, four pinned Git
revisions and all 752 runtime/web files matching the tested wheel. The saved
evidence and offline full audit passed. The research release archive passed
integrity and measurement-reproduction checks. All seven public assets matched
the reviewed checksums after unauthenticated download.

These packaging checks do not add a new performance or quality claim. Future
runtime or dependency changes require validation appropriate to their scope.
No Hugging Face blog, social post, upstream PR, fork default-branch change or
replacement app/Homebrew release was included.
