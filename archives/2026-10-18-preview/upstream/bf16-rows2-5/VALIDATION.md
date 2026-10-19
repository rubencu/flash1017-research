# Validation still required before upstream submission

The recorded checks verify this packet's bytes, application and Python syntax. They do not execute MLX or establish numerical equality, performance, compatibility with all models, or production readiness. No mocked or copied implementation is presented as a passing production test.

Keep the following scope fixed. Do not add unrelated optimizations or change the kernel while searching for a larger result. If a defect requires a code change, retain this packet and assign the changed candidate a new identity before rerunning affected checks.

## 1. Reproduce the actual build

- [ ] Build untouched pinned main and the same main plus this one-file patch with the same locked Python/MLX environment and appropriate native dependencies.
- [ ] Record the actual imported `head.py`, dependency versions and native build identities. Do not substitute the copied application or another experimental overlay.
- [ ] Verify that the changed production `project_logits` route runs for the measured multi-row calls. Importing the module alone is insufficient.

## 2. Check the production projection

- [ ] Compile and execute the new Metal kernel on the intended Apple GPU.
- [ ] Compare actual production outputs for widths **2, 3, 4 and 5** against untouched main's specialized kernel using representative dimensions/dtypes and saved checkpoint activations. Record exact mismatch counts and maximum differences; the intended contract is unchanged shape-matched arithmetic.
- [ ] Check the supported leading-dimension layouts and output shapes through the production entry point.
- [ ] Exercise one-row calls and unsupported cases through both versions: CPU, non-BF16 weights, width below the threshold or not divisible by four, small vocabulary, and six-or-more query rows. Confirm matching fallback behavior, including errors where the original input is invalid.
- [ ] Add focused regression tests beside the upstream head tests, exercising the real implementation rather than a copied formula or fake MLX object. Run those tests and the default project suite. Record existing unrelated baseline failures separately if encountered.

## 3. Measure the one change

- [ ] Freeze a small paired protocol before execution: exact checkpoint revision, matched input tokens, output policy, controller/depth setting, seeds, warmups, cache state and order. Keep reference rounding and every other local feature out of both current-main arms.
- [ ] Measure the projection for supported widths and a bounded paired real-model workload through the production route. Include observed regressions and variation, not only a favorable average.
- [ ] Record generated token IDs, actual stop reasons and activation evidence. Do not treat post-EOS computation or unequal output work as successful-answer latency.
- [ ] Report hardware, commit/build identities, run count and matched settings. State the model and machine limits. A narrow width/matrix test does not demonstrate an end-to-end speedup by itself.

## 4. Review the final submission

- [ ] Keep the diff limited to BF16 rows two through five and necessary production tests. Six-row extension and other features remain separate.
- [ ] Retain the MIT notice and clear prior-implementation credit.
- [ ] Replace the draft's pending status only with results from the actual submitted code. Historical local head results do not satisfy this gate.
- [ ] Present the concrete diff, results and final PR text for the user's review before any public push or submission.

The [current oMLX contribution guide](https://github.com/jundot/omlx/blob/165be1ff5933a020f2f5a7544e1e95a2797f777f/docs/CONTRIBUTING.md) requests relevant production tests and main-based performance evidence. Completing this checklist is future bounded validation, not work performed by this research-release preparation.
