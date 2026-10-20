# Published distribution and upstream plan

The owner approved publication, and both GitHub prereleases were published on
2026-10-19. The [publication receipt](PUBLISHED.json) records their public URLs,
reviewed source commits and verified asset checksums.

| Destination | Published artifact |
|---|---|
| [rubencu/omlx](https://github.com/rubencu/omlx/tree/flash1017-preview) | Branch `flash1017-preview`; [experimental release `v0.7.1rc1-flash1017.2`](https://github.com/rubencu/omlx/releases/tag/v0.7.1rc1-flash1017.2). |
| [rubencu/flash1017-research](https://github.com/rubencu/flash1017-research) | Public repository, branch `main`; [research prerelease `v0.1.0`](https://github.com/rubencu/flash1017-research/releases/tag/v0.1.0). |

Both releases are prereleases. The runtime fork's default branch remains
`main`; the optimized source lives on `flash1017-preview`. Its existing branches
are unchanged. Hugging Face articles, social posts and upstream submissions
are separate decisions and have not been published as part of this release.
Model weights are unchanged, so no Hugging Face model repository is needed.

## Runtime release

The release attaches the exact measured CPython 3.12 Apple Silicon wheel,
its `.sha256` sidecar, `flash1017-2-constraints.txt`, `SHA256SUMS` and
`RELEASE-MANIFEST.json`. The checksum list covers the wheel and constraints
used by the installation guide. The wheel installs the server and web UI.

Measured runtime source: `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`.
Release source: `87b5ede21ec3ac08c3914374077d89857835a693`.
Wheel SHA-256: `0fbdb77ca86ce35b4dca0275b534c1fafde7d70f0f7f6a89d5f964821f7f7f06`.
The release source adds documentation, dependency constraints and workflow
guards; runtime files are unchanged. The inherited wheel-build and Homebrew
release jobs both completed as skipped, preserving the measured binary.

The [runtime release notes](https://github.com/rubencu/omlx/blob/v0.7.1rc1-flash1017.2/docs/releases/flash1017-2.md)
and installation guide identify the pinned checkpoint, 256 GiB M5 Ultra,
single-request profile and measurement scope. Runtime or dependency changes
need their own validation before inheriting these measurements.

## Research release

The research tag identifies commit `297c200f0a06a541c329d24dbe0c163d0f59d515`.
Its assets are `flash1017-research-v0.1.0.tar.gz` and `SHA256SUMS`. The complete
archive contains methods, raw evidence, analysis code, attribution, source
archives and corpus licenses. It excludes weights, environments, caches,
private server settings and wheel binaries. The archive's SHA-256 is
`67e3b25520dd1fc465ce743ec48655662f90dc1feebcd613a88755129f5763da`.

The [research release notes](../releases/v0.1.0.md) explain how to recalculate
results offline. The [fresh-install record](../../validation/release-install-01/README.md)
checks the constrained installation without adding an inference measurement.
All seven public release assets were downloaded without authentication and
matched their reviewed sizes and SHA-256 checksums. The release tags and assets
remain fixed; this branch also records publication status.

## Narrow upstream contributions

Each proposed change needs a branch based on the target upstream revision and
validation against that revision. The measured public-release comparison is
not evidence for the performance of moving upstream `main`.

1. **BF16 head weight reuse for rows 2–5.** Preserve stock per-row arithmetic
   and dispatch limits. Supply real Metal parity checks and a focused
   before/after measurement. Keep row six and unsupported shapes on stock paths.
2. **Prefetch budgeting and lifecycle.** Separate instance-local settings,
   resource ownership and shutdown fixes from any default-budget change.
   Reproduce the issue and add a focused regression check.
3. **Row-cache integration.** Discuss scope with the maintainer before a larger
   patch. Preserve the cache's upstream attribution, quantify its retained and
   temporary memory, and test failure, fallback and unload paths separately.

Do not submit the entire profile or research archive as a single upstream PR.
The fork remains the distribution for the complete measured configuration.
No issue, PR, public message or remote change is authorized by this document.
