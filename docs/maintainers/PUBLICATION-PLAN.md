# Publication and upstream plan

These publication actions remain pending owner approval. The measured findings
are documented in the [article](../ARTICLE.md), [performance methods](../RESULTS.md)
and [quality evaluation](../QUALITY.md).

## Release artifacts

| Destination | Artifact |
|---|---|
| `rubencu/omlx` | Branch `flash1017-preview`; prerelease tag `v0.7.1rc1-flash1017.2`; tested wheel, checksum, dependency constraints and release manifest. |
| `rubencu/flash1017-research` | Public repository, branch `main`; prerelease tag `v0.1.0`; research article, methods, raw evidence, reproducible analysis, attribution and licenses. |
| Optional Hugging Face blog | An article linking the runtime and evidence releases. |

Model weights are unchanged, so no Hugging Face model repository is needed.
The wheel installs the oMLX server and web interface; it does not replace the
macOS menu-bar application.

## Runtime release

Push the reviewed `flash1017-preview` branch to the existing fork, retaining
`upstream` as `jundot/omlx`. Create an explicitly experimental GitHub prerelease
and attach the preserved CPython 3.12 Apple Silicon wheel plus checksum.
Identify measured source commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1` and
wheel SHA-256
`0fbdb77ca86ce35b4dca0275b534c1fafde7d70f0f7f6a89d5f964821f7f7f06`.
Documentation-only branch commits do not change that measured binary.

The prepared branch also adds the recorded dependency constraints and excludes
Flash1017 release tags from the inherited wheel-build/upload and Homebrew
update jobs. These changes do not alter runtime modules. Publish the preserved
wheel, not a new build from the release tag. Keep the fork's existing default
branch and unrelated branches unchanged.

Use the [prepared runtime release notes](https://github.com/rubencu/omlx/blob/v0.7.1rc1-flash1017.2/docs/releases/flash1017-2.md)
as the release body. Attach exactly the reviewed wheel, its `.sha256` sidecar,
`flash1017-2-constraints.txt`, `SHA256SUMS` and `RELEASE-MANIFEST.json`.
`SHA256SUMS` covers the wheel and constraints used by the install guide.

The release description should state the tested checkpoint, 256 GiB M5 Ultra,
single-request profile and installation procedure. Link the evidence behind
7.9% higher generation throughput, 4.5% lower first-text latency and 618 exact
response pairs. Do not generalize to all Macs, models or prompts.

Users can install the wheel in a separate environment and launch
`omlx flash1017 --model /path/to/the/checkpoint`. Runtime and dependency changes
need their own versioned validation before inheriting any measured claim.

## Research release

Create the research repository after review and push its reviewed `main`.
Include the quality and performance bundles, source archives, licenses and
attribution. Keep model weights, Python environments, caches and private
server settings out of the repository. Wheel binaries belong to runtime
release assets; the research bundle records their hashes.

Link the precise runtime artifact from the README and article. Verify the
saved evidence and offline reproduction commands before tagging the research
release. An optional Hugging Face blog should summarize the same scoped
result and link the full evidence.

Use the [prepared research release notes](../releases/v0.1.0.md) as its body.
Attach `flash1017-research-v0.1.0.tar.gz` and its `SHA256SUMS`. The archive
contains the complete repository, including corpus licenses and historical
evidence. The [fresh-install record](../../validation/release-install-01/README.md)
documents the constrained wheel installation; it is not another inference run.

Before any remote writes, verify the reviewed source commits and all staged
asset hashes. Push only the named branches and tags, without force. Create
draft prereleases, upload and verify assets, then publish both as prereleases
without marking either as the latest stable release. Finally, check the public
download links and downloaded checksums. GitHub download links in the prepared
reader guides are publication targets; they are not live until this sequence
has been approved and completed.

The GitHub repositories and releases are the scope of this publication. A
Hugging Face blog, social posts and upstream submissions remain separate
decisions.

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
