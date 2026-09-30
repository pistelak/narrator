# Documentation

Background for changing the library, not for using it — start with the
[top-level README](../README.md) for that.

| Doc | What it is | Read when |
|---|---|---|
| [`engine-comparison.md`](engine-comparison.md) | Why Higgs Audio v3: Higgs against Supertonic on a Czech and code-switched test set, with the cost, intelligibility and naturalness evidence, and the adoption decision. The wider five-engine field is in [`../bench/RESULTS.md`](../bench/RESULTS.md). | evaluating or swapping an engine |
| [`long-form.md`](long-form.md) | The original implementation plan: the measured failure modes of long-form neural TTS and the guard proposed against each, with numbers. Preserved as written; its status table lists where the shipped library differs. | changing chunking, retry, verification or mastering thresholds |
| [`../bench/`](../bench/README.md) | The harness those numbers came from. Re-run it in place when evaluating a new engine — same inputs, same round-trip, comparable results. | measuring anything new |

Findings carry evidence tags: **[measured]** (benchmark, controlled study, or
verified on this machine), **[shipping]** (a real tool's production constant, not
benchmarked), **[inference]** (reasoning from a mechanism), **[folklore]**
(repeated in the wild, no evidence found). Several claims in these documents were
corrected or withdrawn after adversarial review, and the corrections are recorded
rather than quietly applied.
