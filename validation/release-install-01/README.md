# Tested release installation

The preserved `0.7.1rc1+flash1017.2` wheel was installed into a fresh virtual
environment using `flash1017-2-constraints.txt`. All **110 dependency versions**
matched the performance-run receipt, including the same CPython 3.12.15
interpreter and all four pinned Git dependency revisions. No additional
packages were present. `pip check` passed, all **752 runtime and web-interface
files** matched the measured wheel, and the CLI reported the expected version.

The checkpoint metadata check passed without loading weights. This is an
installation check on the original Mac, not an independent reproduction,
new inference run or additional quality/performance measurement. Network
access and the local download cache were available. The constraints pin
versions; they do not hash-lock dependency archives or their build tools.

To verify another installation, run from this research repository with that
environment's Python (replace `/path/to/env` and `/path/to/wheel.whl`):

```sh
/path/to/env/bin/python -I -B tools/verify_release_install.py \
  --wheel /path/to/wheel.whl \
  --constraints validation/release-install-01/flash1017-2-constraints.txt \
  --output /tmp/flash1017-installation.json
```

The verifier reads package metadata and installed files and runs `pip check`.
It does not import inference code or load a model. Its source hash is recorded
in `installation.json`. The measured environment source is
`../performance-v2-01/qualified/block-0-candidate/receipt.json`. The
`checkpoint-check.json` file replaces the local model directory with a label;
all metadata-check fields are retained.
