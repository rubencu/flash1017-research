# Runtime and research repositories

This project provides an oMLX runtime fork and the evidence for its performance
on DeepSeek-V4.1-Flash-1017. The model weights and quantization are unchanged.

| Task | Where to go |
|---|---|
| Install the measured server | The [experimental runtime release](https://github.com/rubencu/omlx/releases/tag/v0.7.1rc1-flash1017.2) and [installation guide](https://github.com/rubencu/omlx/blob/v0.7.1rc1-flash1017.2/docs/flash1017-install.md). |
| Build or modify the server | The [`flash1017-preview` branch](https://github.com/rubencu/omlx/tree/flash1017-preview) of the oMLX fork. |
| Understand the optimizations and results | The [research article](ARTICLE.md). |
| Inspect the evaluation method | [Performance methods](RESULTS.md) and [answer quality](QUALITY.md). |
| Recalculate results or inspect raw evidence | The [evidence guide](../validation/README.md). |
| Check sources and licenses | [Attribution](ATTRIBUTION.md). |

This repository's [v0.1.0 release](https://github.com/rubencu/flash1017-research/releases/tag/v0.1.0)
collects the research and evidence. The runtime release supplies the exact
tested wheel, checksums and a constraints file for its 110 recorded dependency
versions. The runtime installs a server and web UI. It does not replace the
macOS menu-bar application. No additional model repository or weight conversion
is needed.

## Identify the measured build

The evaluated package is **`0.7.1rc1+flash1017.2`**, from source commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. Its comparison baseline is public
oMLX `v0.7.1rc1`, commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`.

The [build identity](../validation/quality-v2-01/BUILD-IDENTITY.json) records
both wheel hashes, source revisions and the pinned checkpoint. The
[performance bundle](../validation/performance-v2-01/README.md) binds those
builds to the timing measurements; the
[quality bundle](../validation/quality-v2-01/README.md) binds them to the 618
paired responses. Documentation may change independently of the measured
runtime. A version label alone is not a replacement for these hashes.

## Contributing

Runtime fixes and installation improvements belong in the oMLX fork. Analysis,
documentation and independent reproductions belong in the research repository.
Each reproduction should identify its source, dependencies, hardware, protocol
and raw observations. The supplied bundles support offline inspection without
the model; repeating inference requires the pinned checkpoint and suitable
hardware.

For source records beyond the measured build, consult
[evidence provenance](EVIDENCE-PROVENANCE.md). Recorded artifacts retain their
execution-time identities and are not installation instructions.
