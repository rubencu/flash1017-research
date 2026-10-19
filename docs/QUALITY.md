# Answer quality: evidence and release requirement

**Status: incomplete for the installable fork.** The 20.06% time-to-first-token
result measures the benchmark prototype's latency. It does not establish that
the complete configuration preserves the baseline's answer quality. A release
claim of faster inference without quality loss requires additional evidence.

## What the existing checks show

The local coding check used eight authored Python tasks, with 160 test cases
per task, greedy generation and fixed prompts. Its
[grades](../archives/2026-10-18-preview/evidence/historical/results/quality-grades-01.json)
and [configuration plan](../archives/2026-10-18-preview/evidence/historical/plans/quality.json)
show:

| Configuration | Tasks passed |
|---|---:|
| Stock oMLX, fixed speculative depth 2 | 8/8 |
| Reference attention rounding, fixed depth 2 | 7/8 |
| Reference rounding plus head reuse, fixed depth 2 | 7/8 |
| Reference rounding plus head reuse through six rows, adaptive depth up to 5 | 8/8 |
| Reference rounding, head reuse through six rows and confidence scheduling | 8/8 |

The two failures were the same task, `longest_unique`: the generated code moved
its sliding-window start backwards when encountering a repeated character.
That program failed 82 of the task's 160 cases. The rounding-enabled depth-2
configuration therefore has an observed task regression against its stock
comparator. This small experiment does not establish a general effect of
rounding on model quality.

A [16K-context check](../archives/2026-10-18-preview/evidence/historical/results/deployment-quality-grades-01.json)
using the same eight tasks passed 8/8 for both the rounding/head depth-2
configuration and the confidence/head configuration. It did not include the
unmodified stock comparator. Neither coding check evaluates the complete
installable fork with all replay, caching and serving changes enabled.

Other evidence answers narrower questions:

- The head-kernel comparison matched all 10,240 compared output-token positions
  against a reference that also enabled attention rounding.
- The replay/cache comparison matched all 1,536 compared output-token positions
  against a reference that also enabled rounding, head reuse and confidence
  scheduling.
- The installed wheel passed functional serving checks. Its short arithmetic
  answer and code-summary responses were not a standardized quality evaluation.

The [methods document](RESULTS.md) defines those component comparisons. Their
matching outputs support the tested paths, not all prompts or the combined
configuration against stock oMLX. The large software-test count also does not
measure model answer quality.

## Why this needs more than a wording change

Some changes aim to preserve computation: reducing redundant weight reads,
reusing table rows and skipping decoder work outside the required dependencies.
Their implementation still needs numerical and state validation.

Two included choices explicitly differ from the stock arithmetic: the
attention-output rounding boundary and the six-row vocabulary projection,
which replaces a matrix-multiply fallback with single-row-style reductions.
Unchanged weights therefore do not imply unchanged model outputs. The complete
performance comparison also changes speculative scheduling and chunk sizes.
A timing measurement alone cannot establish that these combined choices retain
the baseline's capabilities.

Finite quality benchmarks cannot prove equal quality for every possible prompt.
Exact numerical/state agreement on supported paths and a sound argument for
preserving computation offer stronger evidence for an execution optimization.
Where behavior differs, a paired task evaluation must measure the consequences;
absence of a statistically significant difference is not itself evidence of
equivalence.

## Validation required before a no-quality-loss performance claim

1. **Define the two builds and hold evaluation settings fixed.** Compare the
   actual fork wheel with its unchanged public oMLX release base, using the
   same native dependencies, checkpoint, prompt template, sampling settings,
   context limits and output budgets. Record exact identities. Allow normal
   end-of-sequence stopping and count failed or truncated responses.
2. **Separate numerical-policy choices from execution optimizations.** Compare
   logits on identical token prefixes, greedy outputs and relevant target/draft
   state, with boundary cases for the changed kernels and replay paths.
   Investigate mismatches component by component. Choices that change the
   target arithmetic need explicit justification and validation; they cannot
   inherit a claim of unchanged computation from other components.
3. **Evaluate the complete build on public task suites.** Use
   [EvalPlus HumanEval+ and MBPP+](https://github.com/evalplus/evalplus) for coding,
   with fixed dataset/evaluator revisions, and predefined reasoning and
   long-context tasks appropriate to the intended use. The eight authored
   tasks remain regression fixtures, not substitutes for those evaluations.
   Evaluate generated code in an isolated environment.
4. **Fix acceptance rules before collecting outcomes.** Report paired task
   outcomes, baseline-to-candidate regressions and improvements, uncertainty,
   and stopping/truncation behavior. Set any statistical non-inferiority margin
   explicitly before the run; a nonzero allowance cannot be described as proof
   of identical quality. Do not drop failed tasks or adjust the rule to obtain
   a passing conclusion.
5. **Measure speed on the configuration that satisfies those rules.** If a
   numerical-policy feature must be removed or changed, rerun the combined
   performance comparison. The prototype's 20.06% figure cannot be carried
   over to that revised configuration without measurement.

The goal is an installable optimization with a defensible correctness argument,
credible evidence against quality regression and a speed result for that exact
build. Until those conditions are met, the combined speed/quality claim is
unresolved. The bounded component findings remain available for inspection.
