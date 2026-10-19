# Flash1017 research

**Local preparation for `rubencu/flash1017-research`. This GitHub repository has
not been created or published.** This repository holds the research record for
DeepSeek-V4.1-Flash-1017 in oMLX: measured results, raw evidence, analysis tools,
articles, attribution and immutable source snapshots.

The companion **[rubencu/omlx](https://github.com/rubencu/omlx) fork is the canonical runtime implementation** for
continued development and user installation. In the local workspace it is the
sibling `../omlx`. Its installation and validation instructions belong
there. This repository does not maintain a second editable inference runtime.

The October 18 evidence is about a historical prototype. **Its performance
figures do not yet establish the performance or qualification of the current
fork.** [`RUNTIME-LINK.json`](RUNTIME-LINK.json) binds the separate installable
preview to its exact source commit and wheel. Its own validation is recorded
under [`validation/runtime-preview-01`](validation/runtime-preview-01/).

## Read or reproduce the study

- [Complete historical snapshot](archives/2026-10-18-preview/README.md), including source, evidence, analyzer, licenses and the original package manifest.
- [Current article draft](docs/ARTICLE.md) and [complete historical comparison](archives/2026-10-18-preview/evidence/overall/review/report.txt).
- [Results and their limits](docs/RESULTS.md).
- [Repository roles and future runtime bindings](docs/REPOSITORIES.md).
- [Source/dependency boundaries](archives/2026-10-18-preview/DEPENDENCIES.md) and [attribution](docs/ATTRIBUTION.md).
- [Review before publication](docs/REVIEW.md).
- [Concrete publication and upstream plan](docs/PUBLICATION-PLAN.md).
- [Local preparation checks](validation/README.md).

With Python 3.9 or newer, from this repository:

```sh
python3 -I -B tools/verify_research.py
python3 -I -B tools/verify_runtime_preview.py
python3 -I -B archives/2026-10-18-preview/analysis/analyze.py
python3 -I -B -m unittest discover -s archives/2026-10-18-preview/analysis/tests -v
```

These commands verify archived bytes and recalculate saved observations; they
do not load model weights, execute inference, install software or contact a
server. The source patch can also be checked offline using the system `patch`
utility:

```sh
python3 -I -B archives/2026-10-18-preview/source/verify_source_kit.py
```

The archive remains byte-identical to the reviewed preview. Its internal
`PACKAGE-MANIFEST.json` describes that snapshot only. Root-level navigation and
new documentation are separate; [`ARCHIVE-INDEX.json`](ARCHIVE-INDEX.json) and
[`ARCHIVE-SHA256SUMS`](ARCHIVE-SHA256SUMS) bind the retained snapshot. Checksums
provide integrity checking, not a publisher signature or external replication.

## Historical result

On one 80-GPU-core M5 Ultra with 256 GiB memory, the complete prototype reduced
equal-case geometric-mean time to first token from **19.554 to 15.631 seconds:
20.06% less time**, against installed oMLX 0.7.1rc1 with native adaptive MTP.
The fixed suite used the existing mlx-community oQ2.5e-mtp checkpoint, eight
cases, two process blocks and three configurations: 48 measured requests and
six excluded warmups. Current upstream main and the new development fork were
not benchmarked by this study.

![Historical time to first token](archives/2026-10-18-preview/evidence/overall/review/figures/first-token.png)

The models stayed loaded and OS caches persisted; measurements exclude loading
and HTTP transport. The 64K CPython case regressed 2.32%, and sampled peak
footprint rose from 208.35 to 216.10 GiB. Nine requests reached EOS early, so the
original fixed-512 useful-answer qualification failed. First-token timing was
qualified separately. The 128-token supplement is explicitly post hoc. Outputs
differed, and answer-quality equivalence was not established.

The earlier 40.3% replay/cache result uses a different, locally corrected
reference. It is retained as a separate study, not added to the overall result.
The archive also retains failed and undeployed experiments. Offline long-context
equality does not qualify optimized million-token chat. Read [the full scope](docs/RESULTS.md)
before reusing any figure.

## License and credits

Original additions and repository documentation use [Apache-2.0](LICENSE).
Existing per-file MIT/Apache notices and fixture licenses remain applicable;
see [NOTICE](NOTICE) and [attribution](docs/ATTRIBUTION.md). Model weights are
not included. This is credited integration, numerical repair and validation
work; it does not claim a new model, quantizer or inference engine.

Historical documents inside the archive still describe the publication plan
that existed when they were frozen. The current two-repository arrangement is
defined here and in [REPOSITORIES.md](docs/REPOSITORIES.md). No public push,
release, article or upstream submission is authorized merely by this preparation.
