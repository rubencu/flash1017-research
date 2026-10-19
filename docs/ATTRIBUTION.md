# Attribution and licenses

The [complete component attribution](../archives/2026-10-18-preview/docs/ATTRIBUTION.md)
is retained with the historical source. This repository preserves its original
notices and fixture licenses; the source snapshots are not blanket relicensed.

DeepSeek supplies the model and DSpark; mlx-community supplies the unchanged
quantized checkpoint; Apple MLX and MLX-LM/oMLX supply the inference foundation.
The work adapts existing approaches and contributes concrete numerical repairs,
integration, lifecycle handling and exact-checkpoint measurements.

HotRows is an established verbatim copy from tacos8me/m5-ultra, with retained
Apache-2.0 attribution. The local ownership, reader coordination and cleanup
adapter is distinct. Analogous earlier BF16 row reuse and frozen confidence-cost
policies from tacos8me, decoder replay and arithmetic partition preservation
from ds4 and related MLX work, and gnahzhang's reference-rounding documentation
are credited as related work. Similarity alone does not establish direct
derivation or independent invention. Do not claim the general ideas as new.

Original additions and new repository documentation are offered under the root
[Apache-2.0 license](../LICENSE). Existing per-file MIT and Apache-2.0 licenses
remain applicable. The [fixture notices](../archives/2026-10-18-preview/evidence/overall/fixtures/NOTICE)
cover CPython, curl and retained Project Gutenberg materials separately. Model
weights and a copied application distribution are not included.

The historical audit records describe local agent/code-path checks, not external
peer review or independent-machine replication. Preserve that distinction in
future articles, releases and upstream submissions.
