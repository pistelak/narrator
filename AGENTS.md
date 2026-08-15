# Working on narrator

Verified long-form TTS. The pipeline is `chunk → synthesize/verify (ASR
round-trip), retrying or sentence-splitting as needed → stitch → master`; by
default a render that cannot verify every chunk raises instead of writing a
file. That refusal is the product — do not weaken it to make something pass.

## Commands

- Tests: `.venv/bin/python -m pytest tests/ -q` (fast, no models; the fake
  backend drives the real pipeline)
- Lint: `.venv/bin/python -m ruff check narrator/ tests/ bench/` (rule
  selection is pinned in pyproject — do not rely on ambient config)
- Dev setup, if `.venv` is absent:
  `python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'`
- Real-model setup (Apple Silicon):
  `python3.12 -m venv .venv-higgs && .venv-higgs/bin/pip install -e '.[higgs,parakeet]'`

## Workflow

- **Orchestrate broad reviews; iterate fixes directly.** For reviews, audits,
  and multi-file exploration, split the work into distinct angles; fan them
  out to subagents when delegation is available, otherwise cover them
  sequentially or lean on the independent review below. Verify findings
  adversarially before acting on them. For surgical fixes in the
  measure → fix → test loop, work directly — the context accumulated across
  iterations is the asset, and indirection loses it.
- Suite and lint green before every commit; a verifier change is not done
  without a test pinned to the data that motivated it.
- **Before pushing anything beyond a trivial fix, get an independent review**
  (next section), effort scaled to the change. Verify each finding against
  the code before acting; fix what is real, name what you skip.
- Scale the landing to the change: small changes with the suite green may
  land on `main` directly after the review gate. Substantive changes —
  verifier semantics, public API, new features — go via a branch and pull
  request so CI gates the merge on both platforms and a human sees the diff
  before it lands.

## Independent review

Use a capable reviewer from a different model family than the authoring
agent (any independent model for human-authored changes). The Codex CLI
example below therefore fits changes authored outside the GPT family; for
Codex-authored changes use a non-GPT equivalent, and if none is available,
say plainly that the independent-review requirement is unmet:

```bash
P=$(mktemp); O=$(mktemp); cat >"$P" <<'EOF'
<goal, exact paths in scope, constraints, non-goals,
 proof expected per claim, output shape>
EOF
codex exec -s read-only -C . \
  -m gpt-5.6-sol -c model_reasoning_effort="xhigh" \
  -o "$O" - <"$P"
```

The prompt contract does the work: state the goal, the exact scope, what is
out of scope, and demand file:line evidence for every claim. Read the output
file and verify every finding against the code. In review-only tasks, report
verified findings without editing; when implementation is in scope, fix
verified findings and name any left unresolved.

## Load-bearing rules

- **Comments are evidence.** Docstrings and comments carry the measured
  failure that motivated each rule. Never strip them; when changing behavior,
  update the measurement story too.
- **Every verifier change needs a test pinned to real data** — the suite is a
  measured-regression suite. A new fold or rule without the transcript that
  motivated it is not done.
- **No project vocabulary in the library.** `fold()` holds phonology; the
  letter-name and numeral tables are language data. Word-specific
  equivalences come from callers via `sound_alikes` (derived from their
  pronunciation lexicon). A regression test pins the motivating vocabulary
  boundary.
- **The pronunciation lexicon applies at synthesis only**; verification always
  compares against the caller's original text. Substituting upstream makes
  correct audio fail.
- **`source_rate` must be the backend's actual rate** everywhere an ASR is
  constructed — a mismatch silently corrupts every verdict. `default_verifier`
  exists so callers stop assembling this by hand; keep it the single policy.
  Note a backend may only settle its true rate during its first synthesis,
  which is why `render()` builds the default verifier on first use, never
  eagerly.
- The input vocabulary is `Text` and `Gap` only. Narrator never learns markup;
  callers translate their own conventions into segments.
- Verification policy: with the `[parakeet]` extra, `default_verifier` runs
  the fast recogniser on every chunk and consults Whisper only on rejection —
  either may confirm; without the extra it is Whisper alone. Cascade order
  affects cost, never accept-if-any semantics.

## Layout

`narrator/`: `chunking` (sentence boundary lives here — one home, three
importers), `synth` (retry ladder, cheap checks, pronunciation + acronym
spelling), `verify` (coverage scoring, folds, hard-fail rules, verifiers),
`render` (orchestration, quarantine), `audio` (DSP/mastering), `asr`
(recognisers), `backends/` (TTS engines + deterministic fake), `types`
(protocols), `cli` (the `narrate` entry point), `__init__` (the public API —
new exports go in `__all__`). `bench/` is measurement tooling; its results
justify the design.
