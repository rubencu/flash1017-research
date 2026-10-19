Recalculate the saved Flash-1017 results

Requirements: Python 3.9 or newer. Standard library only. This works without
Apple Silicon, model weights, MLX, a server, an internet connection, or the
original laboratory directory. It reads only packaged JSON evidence. Recorded
absolute source paths are provenance strings and are never opened or executed.

From the extracted package directory:

    python3 -I analysis/analyze.py

Write both outputs (choose new filenames; overwrites are refused):

    python3 -I analysis/analyze.py --json ../recalculated.json --text ../recalculated.txt

These output paths keep generated reports outside the archive directory.
verify_package.py deliberately reports unmanifested files inside that directory.

Run the CPU contract tests:

    python3 -I -B -m unittest discover -s analysis/tests -v

Defaults are resolved relative to this script, so invocation also works from
another current directory. Explicit overrides are available:

    python3 -I analysis/analyze.py --plan evidence/overall/plan-01.json \
      --raw-dir evidence/overall/raw --review-dir evidence/overall/review

Input contract

The frozen plan's original SHA256 is pinned in the analyzer. It contains the
authoritative input-token vectors, sampling settings, case/seed schedule and
three arm profiles. Exactly these six process files are required:

    00-native2048-block0.jsonl
    01-native8192-block0.jsonl
    02-complete-block0.jsonl
    03-complete-block1.jsonl
    04-native8192-block1.jsonl
    05-native2048-block1.jsonl

The packaged plan, six raw files and two review files must also match their
byte counts and SHA256 entries in EVIDENCE-INVENTORY.json. Original raw hashes
in that mapping must match the archived independent audit. This binds the
normalized public copies without opening any original source path. An alternate
inventory can be supplied with --inventory PATH. As with any unsigned local
archive, agreement with an included manifest is integrity checking, not proof
of a trusted publisher's identity.

The analyzer validates all 54 requests: six excluded warmups and 48 measured
requests. It checks case pairing/order, seeds, input hashes and lengths, output
token lengths/ranges, EOS token 1 and its first position, positive finite timing,
monotone delivery timestamps, timing endpoints and stored-rate arithmetic. It
also checks recorded activation/cleanup, MTP/replay/priming fields, common device,
dependency and memory policies, and process ordering. It refuses incomplete,
duplicated, malformed or inconsistent records instead of selecting a subset.

It calculates all pair/group, per-case and per-arm geometric means used by the
saved supplement; resource summaries; early EOS records; the CPython 64K
regression; and the complete-package memory tradeoff. Numerical results are
compared with the packaged independent-audit.json and supplement.json. Any
disagreement fails the command. Exit code 0 means the observation checks and
saved-review comparison passed; exit code 2 means analysis was refused.

Interpretation

The primary denominator is native2048: installed oMLX 0.7.1rc1 with native
adaptive MTP. This is not a current-main benchmark. The headline is 20.06% less
time to first token, not a general model speed or quality-equivalence claim.

The original fixed-512 qualification remains FAILED because nine requests
reached EOS before their last output position. Full-512 elapsed/decode metrics
remain descriptive forced-continuation observations, including post-EOS work.
Prefill and TTFT qualify separately and contain no post-EOS continuation.

The first-128 analysis is POST HOC and uses saved arrival timestamps from the
original 512-token executions. It is available only if every measured request
has no EOS before zero-based position 128. EOS127 blocks it for the whole suite;
EOS128 permits the first128 positions0–127. No subset or shorter-horizon rescue
is used. These are delivery times, not newly executed128-token requests,
terminal synchronization/cleanup times or isolated GPU timings; speculative
batches may compute beyond the displayed endpoint.

All48 records remain included. Two blocks on one machine do not establish a
confidence interval, other-hardware performance or typical-user workload mix.
Sampled memory peaks may miss brief transients; disk counters are process-wide.

Evidence boundary

This is an independent numerical recalculation of saved observations, not a
new execution, external peer review, portable-runtime qualification or a replay
of the original source/model audit. The original audit also checked local
source pins, launch/exit and service-restoration evidence. Its accepted status
is retained and compared; this portable tool does not access those historical
source locations or prove that recorded events occurred.

Packaged raw files have deterministic path normalization. Their byte hashes
therefore differ from original raw hashes retained in the historical audit.
The report identifies packaged SHA256s without pretending those are original
hashes. The release EVIDENCE-INVENTORY.json records the original/public mapping.
No raw timing, token, EOS, resource, eligibility or result field is transformed.

No chart dependency is required. The reviewed figures and their original
plotting code are preserved elsewhere in the package.
