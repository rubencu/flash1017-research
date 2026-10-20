# Faster DeepSeek-V4.1-Flash-1017 inference on Apple Silicon

This project optimizes **DeepSeek-V4.1-Flash-1017 in oMLX**, an inference server
built on Apple's MLX framework. It reuses vocabulary weights and disk-backed
lookup-table rows without changing the model's weights or quantization.

On an **M5 Ultra Mac Studio with 80 GPU cores and 256 GiB unified memory**, the
measured package delivered **7.9% higher generation throughput and 4.5% less
time to first visible text** across eight workloads. A separate quality
comparison reproduced the public-release baseline's **answer text and finish
reason on all 618 requests**, with zero observed task regressions.

| Comparison with public oMLX `v0.7.1rc1` | Observed result |
|---|---|
| Generation throughput, eight mixed workloads | **7.9% higher**, improving in all 16 workload/block pairs |
| Client first-visible-text latency, the same workloads | **4.5% lower**, improving in 11/16 pairs |
| Six natural-source workloads | **8.0% higher generation throughput; 5.5% lower first-text latency** |
| Separate quality comparison | **618/618 identical response texts and finish reasons; zero regressions** |

The timing results are equal-case geometric means from two process blocks.
First-text latency varied noticeably between blocks, making its smaller gain
less consistent than the generation gain. Only three workloads finished
normally within the 512-token limit; completed-answer latency applies to their
six pairs. The other requests remain in throughput and first-text results.
These measurements describe one machine and fixed workloads, not statistical
significance or performance on other hardware.

## How it works

The runtime combines three changes: sharing BF16 vocabulary-weight reads
across verification rows while preserving the stock arithmetic; caching raw
Engram lookup-table rows in 512 MiB of retained storage; and increasing Engram
prefetch admission from 16 to 64 MiB. Both measured builds use stock attention
behavior, native adaptive multi-token prediction and prompt chunks configured
at no more than 2,048 tokens. The [article](docs/ARTICLE.md) explains how these
changes reduce work and how their output behavior was checked.

## Measured software and model

| Component | Exact identity |
|---|---|
| Optimized package | `0.7.1rc1+flash1017.2`, commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1` |
| Public-release baseline | `v0.7.1rc1`, commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46` |
| Model | [`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab), revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab` |

This is a comparison with the specified public release. Upstream `main` and
other model/checkpoint combinations were not measured. The
[build identity](validation/quality-v2-01/BUILD-IDENTITY.json) and evidence
manifests include the exact wheel and source hashes.

Install the exact tested wheel from the companion
[experimental runtime release](https://github.com/rubencu/omlx/releases/tag/v0.7.1rc1-flash1017.2),
following its [installation guide](https://github.com/rubencu/omlx/blob/v0.7.1rc1-flash1017.2/docs/flash1017-install.md).
This research repository contains the methods, recorded results, analysis code
and attribution. Model weights are not included.

## Read the research

- **[Runtime and installation](docs/REPOSITORIES.md):** where to get the server, build it and find its setup guide.
- **[Article](docs/ARTICLE.md):** how the optimizations work and what the results mean.
- **[Methods and results](docs/RESULTS.md):** timing definitions, workloads, per-case results and limitations.
- **[Answer quality](docs/QUALITY.md):** task selection, executable grading, exact response comparison and scoring limits.
- **[Performance evidence](validation/performance-v2-01/README.md):** raw measurements, frozen sources and offline reproduction.
- **[Quality evidence](validation/quality-v2-01/README.md):** all 618 requests and responses, grades and provenance.
- **[Attribution](docs/ATTRIBUTION.md):** reused implementations, related work and contribution boundaries.

## Inspect and recalculate the evidence

With Python 3.9 or newer, from this repository:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v2-01
python3 -I -B tools/verify_quality_evidence.py --folder validation/performance-v2-01
python3 -I -B validation/performance-v2-01/reproduce.py --mode measurements --output /tmp/flash1017-performance-measurements.json
```

These commands verify saved files and recalculate the performance ratios from
raw measurements. They require no model weights or GPU and execute no model
answers. The [full audit command](validation/performance-v2-01/README.md#reproduce-from-saved-files)
also verifies source/native identities using the two exact wheel files, without
installing them. Repeating inference requires the pinned checkpoint, suitable
hardware and the declared process/cache sequence.

Exact response parity is a finite observation: it does not prove equivalence
for every prompt or sampling setting. The quality evaluation retains four
shared output-budget stops and two documented evaluator exceptions in its
full denominators. Both memory profiles requested the same startup guideline,
but their effective wired limits differed. These qualifications are detailed
in the methods and quality documents. The
[evidence provenance](docs/EVIDENCE-PROVENANCE.md) reference records source
identities and audit details.

## License and credits

Original additions and documentation use [Apache-2.0](LICENSE). Existing per-file
MIT/Apache notices and corpus licenses remain applicable; see [NOTICE](NOTICE)
and [attribution](docs/ATTRIBUTION.md). The contribution is implementation and
integration of execution optimizations, with measurements for the identified
model, runtime and machine.
