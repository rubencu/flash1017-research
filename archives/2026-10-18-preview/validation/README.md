# Local verification records

These are CPU-only local checks performed while preparing the review package. They are not new inference measurements, external replication or qualification of an installer.

- [Analysis relocation](analysis-relocation.json): the saved-results analyzer and its 36 tests passed in a fresh directory. During the analyzer check, reads from the original workspace, networking and model-library imports were blocked by a Python audit hook. The recorded Python version is authoritative.
- [Source kit](source-kit.json): copied/generated file integrity, Python parsing, frozen calibration, recorded source edges and exact five-file patch reconstruction passed, including a relocated copy.
- [First upstream packet](../upstream/bf16-rows2-5/verification.json): clean patch application, Python syntax and preservation checks. No Metal or production model execution.

The original benchmark evidence is under [evidence](../evidence/overall/review/report.txt). These later checks recalculate or inspect it; they do not replace its original qualification or change the failed fixed-512 result.

The final archive has a separate review receipt and checksum alongside it. That receipt covers verification of the actual extracted archive, whose digest cannot be included inside itself.
