# Source kit and dependency boundaries

This is a locally prepared **research source archive for review**, not an
installer. It contains the complete accepted local source contribution and
its associated tools, including code loaded from outside the copied runtime.
It does not contain a complete external inference environment. Nothing in this
archive starts, changes or qualifies the model merely by being downloaded.

The original laboratory files remain unchanged. `SOURCE-INVENTORY.json` maps
each copied file's original path and SHA-256 to its public path and SHA-256.
It records exact path substitutions: the full original workspace prefix becomes
`${ORIGINAL_LAB_ROOT}`, then remaining home-directory prefixes become
`${USER_HOME}`. These are descriptive placeholders, not automatically resolved
configuration variables. The original source/profile hash guards are retained;
they have not been relaxed or rewritten to make the archive launchable.

## What is included

All paths in this table are relative to `source/`, which preserves the original
laboratory layout. A role is recorded for every file in `SOURCE-INVENTORY.json`.

| Contribution or purpose | Included source |
|---|---|
| BF16 vocabulary-head reuse and the explicit six-row path | `outputs/flash1017-lab/flash1017/head.py`, activation in `bench.py`, original profiles and tests |
| Causal confidence policy, calibration procedure and actual frozen values | `flash1017/policy.py`, `calibrate.py`, `patches/confidence-experiment.patch`, `results/confidence-policy6-01.json`, associated plans/tests under `outputs/flash1017-lab/` |
| Attention reference-rounding boundary in target and draft | `outputs/flash1017-lab/patches/reference-arithmetic.patch` and the actual modified upstream files |
| Chunk size, native prefetch budget, memory policy and early runtime hooks | Full `outputs/flash1017-lab/flash1017/` package, including `bench.py`, `memory.py`, `serve.py`, bootstrap, launchers and profiles |
| Decoder replay | `work/cone-experiment/cone.py` and `work/serving-cone-v3/cone_serving.py` |
| Shape/dispatch arithmetic and three-stage draft priming | `work/cone-v2/stabilization.py` and `work/cone-dspark-stable/stabilization.py`; these are the actual AST-extracted definitions, not substitutes |
| Bounded HotRows ownership, reader coordination and cleanup | `work/serving-cache-v3/serving_cache.py`, unchanged `hot_rows_upstream.py`, provenance and upstream license/notice |
| Live qualified server integration | `work/serving-integration-v3/{runtime,evidence}.py`, `work/serving-server-v3/serve_candidate.py`, `work/serving-launch-v3/launch.py`, `work/serving-shortcut-v3/shortcut.py` |
| Overall-comparison integration | Separate `work/serving-integration-v3b/{runtime,evidence}.py`, `work/overall-comparison-01/worker.py`, source-activation audit, freeze/supervisor and original analysis code |
| Historical numerical, serving and fixture instrumentation | Qualification runners, HTTP checkers, analysis scripts, state-audit helpers and tests, explicitly labeled as archival tools |
| Actual upstream source delta | Five changed files under `outputs/flash1017-lab/_runtime/omlx/`, five public baseline files under `upstream-baseline/omlx/`, and `patches/complete-runtime-vs-v0.7.1rc1.patch` |

The archive intentionally includes both integration versions: the server uses
`serving-integration-v3`, whereas the overall comparison uses `v3b`. Omitting one
would omit part of the measured work. Likewise, copying only `_runtime` would
miss the external head/policy modules, AST definitions and scoped adapters.

The archive also includes historical plans and helper tools that are not
accepted runtime features. Their presence does not enable larger-cache,
envelope-padding, whole-request diagnostic or optimized 1M-serving variants.
Historical comments and statuses are retained; the current release report
controls the claims. A copied profile's `qualified` field describes the original
measured environment, not this archive after path normalization or a new build.

## Source audit and patch verification

`source/SOURCE-CLOSURE-AUDIT.json` records a static, no-import inspection from
the shortcut, launcher, server and overall benchmark entrypoints. It records
ordinary imports, literal source references and explicitly reviewed dynamic
edges for the loaders and AST definitions. The audit finds **26 reachable local
source files, with no missing local code targets**, including the five modified
upstream files. It also records external import names, unchanged upstream
dependencies and literal data references. Dynamic execution is not generally
provable from an AST scan; this is not a claim of full external dependency closure.

The copied runtime has exactly five changed Python files relative to the
installed baseline: DeepSeek `config.py`, `language.py`, `dspark.py`, `loading.py`
and shared MTP `batch_generator.py`. Each baseline file was fetched from public
oMLX commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46` and matched byte for byte
to the installed baseline. The combined patch was applied with zero fuzz in a
temporary directory and recreated all five measured source files byte for byte.
The receipt records the base URLs, hashes, patch hash and result. This check
does not compile native code, import inference packages or run a model.

From the package root, the following command checks all source-copy hashes,
Python syntax, recorded code edges, the frozen calibration values and offline
five-file patch reconstruction. It requires ordinary Python and the `patch`
utility; it needs no model, MLX installation, network or original workspace:

```sh
python3 source/verify_source_kit.py
```

The original scripts under `source/outputs/` and `source/work/` are included for
inspection. Some are launchers or historical tools that control a server when
run in their intended environment. They are archived as non-executable files;
the command above only parses or hashes them. It does not run those scripts.

## External runtime that is not bundled

The measured system uses oMLX.app 0.7.1rc1, app bundle build 3310, with embedded
Python 3.11. Its oMLX build marker is `261016042522-macos26-27`. The original
`run.sh` selects the installed app's Python and site-packages, and
`setup_runtime.py` copies validated installed Python source. The native
extensions, Python interpreter, web assets and complete package installation
are not bundled here. No copied application tree, native library, virtual
environment or model weight is included.

The frozen overall comparison records MLX 0.32.4, MLX-LM
0.31.4.dev132+g94cdcae13, MLX-VLM 0.7.8, NumPy 2.3.5 and Transformers 5.17.0.
The source also uses psutil; the inspected installed metadata reports 7.2.2.
`source/EXTERNAL-BUILD-RECEIPT.json` records the observed name/version and metadata
hash for 173 installed distributions and the five expected native-file hashes.
This is an external-environment receipt, not a tested resolution lockfile or
a guarantee that reinstalling these versions recreates the same native build.

The [official historical release](https://github.com/jundot/omlx/releases/tag/v0.7.1rc1)
provides wheels and DMGs. Its public cp311 wheel has SHA-256
`5b1487bd09fd85da6e7566cd94f0e4b30be03277e49a9250811cf05addc9b26e`.
Earlier read-only inspection found all five expected native module paths but
different binary hashes from the measured installation. This establishes
nonidentity, not ABI or behavioral incompatibility. Full DMG/dependency
equivalence, reproducible native toolchain/build flags and a clean installation
have not been established. The relevant public Python baseline is available;
exact reconstruction of the measured complete environment remains unverified.

Most unchanged oMLX code is external. The five-file baseline is sufficient to
verify the patch, but cannot run oMLX on its own. Current upstream main also has
a different native build/import layout; this archive has not been rebased or
qualified against it. No startup hash guard has been bypassed to hide that gap.

## Model, calibration and evidence inputs

The weights are the existing
[mlx-community checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. They are not redistributed.
The copied checkpoint receipt records original content hashes and historical
file fingerprints. Its model path is normalized, and another machine will not
have the same inode/mtime fingerprints. It is provenance, not a portable
authorization to load arbitrary model files.

The fitted `confidence-policy6-01.json` values are included in the source kit
and were checked against the candidate profile. The original calibration trace
is in `evidence/historical/results/cost6-01.jsonl`; the calibration source and
plans show how those values were fitted. The file's original calibration hash
is retained even where the public data copy has redacted paths; the package's
evidence mapping distinguishes the two byte identities.

The complete frozen overall plan contains authoritative input token vectors.
The package's evidence section carries it, raw outputs and timing observations,
and public fixture provenance/licenses under `evidence/overall/fixtures/`.
Those data are distinct from the source tree. Historical scripts may still
refer to their original lab-relative data locations. They have not been changed
to silently retokenize text, relocate evidence or treat omitted data as present.

The original candidate and launcher manifests also pin historical reports,
snapshots and many external files. Their sanitized copies remain inspectable,
but not all guard targets are present in the source tree. Missing external
runtime files, large saved state/logit arrays and unselected historical traces
remain explicit omissions. The source audit does not replay numerical or HTTP
evidence, and the archival launchers should not be mistaken for an install path.

## Attribution and future packaging

Existing per-file MIT/Apache-2.0 notices and upstream license texts are preserved.
HotRows is an established verbatim copy from tacos8me, with its original source
identity and Apache-2.0 attribution. Other analogous prior implementations—BF16
row reuse, frozen confidence/cost policy, decoder replay and arithmetic
partition preservation—are related work; similarity alone establishes neither
direct derivation nor independent invention. The release's current attribution
addendum supplies the newly identified credit without rewriting frozen notices.
Original additions require the explicit release license selected by the owner.

A later installable release needs one verified packaging route, complete
external build/dependency identity, portable model/source receipts and a fresh
installation with finite numerical/fallback and serving checks. Those steps
have not happened here, and no model execution or service changes were made
while preparing this source kit. The saved-data statistics and offline patch
check are usable now; inference reproduction and product qualification remain
separate capabilities.
