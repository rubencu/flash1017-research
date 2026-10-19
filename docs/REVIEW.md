# Maintainer release checklist

This checklist is for the repository owner reviewing a release. Readers of the
research can start with the [article](ARTICLE.md) or [methods](RESULTS.md).

The proposed public destination is **`rubencu/flash1017-research`**. This remote
has not been created. The local repository is on `main`. The runtime fork is the
sibling `../omlx`; its exact source and wheel are bound in `RUNTIME-LINK.json`.
See [PUBLICATION-PLAN.md](PUBLICATION-PLAN.md) for the proposed remote actions.

Before any publication, review:

1. The reader-facing README, article, methods, repository layout and exact runtime identity.
2. The immutable historical source/evidence snapshot and its checksums.
3. The headline's prototype/build scope, failed acceptance checks, regressions,
   memory tradeoff, attribution and licenses.
4. The runtime fork's actual build/install/serving status and any proposed new
   measurement. Performance claims must identify the build that was measured.
5. The exact remote creation, push, release, article and upstream actions to take.

Development notes inside the archive contain superseded release proposals.
They are retained with the experiment record. The repository layout for this
release is defined by the root README and `docs/REPOSITORIES.md`.

The retained narrow upstream patch packet is a historical draft. It is not a
submitted PR, a promise of maintainer acceptance or a current-main performance
result. Any actual upstream branch and submission belong to the runtime fork.

The archive verification commands perform no installation or model execution.
The optional installed-runtime smoke harness starts a real server and loads
the model; its results are recorded separately. Review remains required
before creating the remote, pushing commits or sharing material publicly.
