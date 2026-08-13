# Documentation

| Doc | What it is |
|---|---|
| `engine-comparison.md` | Why Higgs Audio v3. Five engines on a Czech and code-switched test set, with the cost, intelligibility and naturalness evidence, and the adoption decision. |
| `long-form.md` | Why the library is shaped this way. The measured failure modes of long-form neural TTS and the guards against each, with numbers. |
| `../bench/` | The harness those numbers came from. Re-run it in place when evaluating a new engine — same inputs, same round-trip, comparable results. |

Findings carry evidence tags: **[measured]** (benchmark, controlled study, or
verified on this machine), **[shipping]** (a real tool's production constant, not
benchmarked), **[inference]** (reasoning from a mechanism), **[folklore]**
(repeated in the wild, no evidence found). Several claims in these documents were
corrected or withdrawn after adversarial review, and the corrections are recorded
rather than quietly applied.
