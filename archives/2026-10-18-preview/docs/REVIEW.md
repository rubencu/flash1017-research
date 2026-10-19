# Publication review

This is the complete local review copy. No public push, release, article, issue or PR has been authorized or performed as part of preparing it.

## Proposed public release

Use the existing **rubencu/omlx** fork as the canonical home, with a dedicated research branch and immutable tag. Proposed names are `research/flash1017-1017` and `flash1017-research-preview-20261018`; neither has been created by this packaging step. Preserve unrelated fork changes when preparing that branch.

The GitHub release would contain this complete archive and its SHA-256 checksum. The branch would carry the same source, documentation and evidence, preserving the package layout and notices. The [article](ARTICLE.md) would introduce the result and link to that exact release. A Hugging Face blog version could link to the same canonical source and evidence. There is no new model-weight repository or separate engine in this proposal.

The public claim is **20.06% lower geometric-mean time to first token** in the declared eight-case suite against installed oMLX 0.7.1rc1 with native adaptive MTP. It includes all accepted local changes and their configuration. The memory increase, 64K CPython regression, failed original fixed-512 qualification, post-hoc prefix analysis and unestablished answer-quality equivalence stay visible beside that claim.

## What readers can do now

- Recalculate every reported overall comparison from the saved inputs and raw observations using ordinary Python.
- Inspect all accepted local code, its upstream deltas and the numerical/state repairs, with original and redistributed file identities.
- Review the component studies, failures, attribution and exact limits of the comparison.
- Adapt the implementation to an appropriate runtime, or build on the [narrow upstream patch packet](../upstream/bf16-rows2-5/README.md).

Installing this archive into ordinary oMLX is not yet a supported user workflow. The runtime source preserves the historical prototype and depends on an external installed application environment. A future installable release requires a pinned build and bounded installation/inference checks of the actual distributed artifact. Those checks are separate from this evidence release.

## Upstream sequence

First, validate the prepared BF16 rows 2–5 patch against its pinned main build using the [finite checklist](../upstream/bf16-rows2-5/VALIDATION.md), then review the final diff, measurements and PR text before submission. The extracted patch has only CPU packet checks so far. It must not inherit the complete configuration's speedup claim.

Keep the optional reference-rounding contract separate. Consider confidence, replay and storage integration only as coherent, separately verified contributions, coordinating overlapping work. The [upstream plan](UPSTREAM.md) explains the scope of each. No further optimization sweep is part of this release preparation.

## Author review

Review the [article](ARTICLE.md), [release notes](RELEASE-NOTES.md), [attribution and license scope](ATTRIBUTION.md), [dependency boundary](../DEPENDENCIES.md), and proposed destinations above. Original additions and release documentation are proposed under Apache-2.0; preserved third-party terms continue to apply. AI assistance is disclosed in the attribution document.

Publication remains pending the author's approval of the concrete contents and destinations. Upstream submission additionally requires the missing production validation and review of the resulting PR. Any edits after this package is frozen require a new manifest and archive checksum.
