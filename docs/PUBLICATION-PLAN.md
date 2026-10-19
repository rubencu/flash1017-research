# Maintainer publication and upstream plan

This document proposes release actions for the repository owner. The research
findings and their methods are in [RESULTS.md](RESULTS.md).

No action below has been performed. The local source and artifacts are the
reviewable deliverable; publication requires Ruben's approval.

## Quality prerequisite

The package is installable and functionally tested, but a claim of faster
inference without quality loss is not ready. The complete packaged configuration
has not received a paired answer-quality evaluation against its unchanged
release base. The [quality evidence and validation requirement](QUALITY.md)
records the existing task regression, the limits of component parity checks,
and the checks needed before such a claim. A disclaimer does not complete this
validation. Any no-quality-loss release claim must wait for that evidence, and
speed must be measured on the exact configuration that passes.

## Runtime: the existing rubencu/omlx fork

Push the tested `flash1017-preview` branch to the existing fork. It contains the
complete application source and accepted profile, its tests, installation docs
and wheel builder. Keep `upstream` pointed at `jundot/omlx` for future updates.
The existing remote `main` need not be overwritten. Making the preview the
fork's default branch is a separate optional repository-setting change.

Create an explicitly experimental GitHub prerelease from the reviewed commit,
attaching the CPython 3.12 Apple Silicon wheel and its SHA-256 sidecar. Link its
exact source commit, public-native build provenance and functional validation.
The wheel installs the server and web UI; it does not replace the official
menu-bar app. Model weights remain the unchanged pinned mlx-community download.

Users can install the wheel in a separate environment, run
`omlx flash1017 --model /path/to/the/checkpoint`, and use the web chat or usual
oMLX API clients. Contributors can edit this fork, rebuild it and submit fixes
to it. Keep future runtime work here, with new versioned validation when the
inference code or native dependencies change.

The 20.06% result must identify the benchmark prototype and its installed-release reference.
A functional smoke check does not make it a benchmark of this wheel or
establish answer-quality equivalence. Do not turn that result into a general
claim about all models or all Macs.

## Research: a new rubencu/flash1017-research repository

Create this repository only after review, then push its local `main`. It holds
the immutable experiment record, raw observations, analysis, figures, article,
attribution and separate validation of the installable fork. `RUNTIME-LINK.json`
binds the current reviewed source and wheel; the archive keeps the identity of
the code that actually produced its measurements.

Readers can check source and evidence, recompute the saved results, inspect
failed experiments, reproduce the protocol on their hardware, and propose
analysis corrections. The repository also contains an optional real-serving
smoke harness; unlike the archive verifier, it loads the large model and sends
requests through the installed fork.

The article can be published on GitHub or as a Hugging Face blog post after
review. Link both repositories and the exact result scope. A new Hugging Face
model repository would duplicate unchanged weights and is unnecessary.

## Focused upstream work after the fork is reviewable

Prepare each candidate on a fresh branch from then-current upstream `main`.
Keep the complete experimental distribution in our fork; do not submit its
6,000-plus-line profile, research archive or release helper as one upstream PR.

1. **BF16 head reuse for rows 2–5:** retain the existing arithmetic and dispatch
   boundary. Supply actual Metal parity tests and a fresh same-machine before/
   after measurement against current main. Treat row 6 separately because its
   old fallback uses different reduction arithmetic.
2. **Numerical and DSpark state repairs:** first reproduce each defect on
   current main through the production path. Submit the smallest correction
   with a failing regression and representative model validation. Do not call
   an opt-in numerical-policy change a universal correctness fix.
3. **Prefetch/cache lifecycle improvements:** separate per-instance budgeting
   and cleanup from the full row-cache subsystem. Preserve default behavior;
   document memory cost and measured benefit for any proposed default change.
4. **Larger replay or confidence features:** discuss scope with the maintainer
   before a large PR, in accordance with the contribution guide. Keep them
   opt-in in our fork until the implementation and evidence justify broader use.

The archived BF16 patch packet is a starting reference, not a ready-to-submit
current-main PR. No upstream issue, discussion, PR, push or public message is
authorized by this plan itself.
