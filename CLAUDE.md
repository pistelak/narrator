# Working on narrator

Verified long-form TTS. The pipeline is `chunk → synthesize → verify (ASR
round-trip) → retry ladder → stitch → master`; by default a render that cannot
verify every chunk raises instead of writing a file. That refusal is the
product — do not weaken it to make something pass.

## Commands

- Tests: `.venv/bin/python -m pytest tests/ -q` (fast, no models; the fake
  backend drives the real pipeline)
- Lint: `.venv/bin/python -m ruff check narrator/ tests/ bench/` (rule
  selection is pinned in pyproject — do not rely on ambient config)
- Real models (Apple Silicon): `.venv-higgs/bin/python` has `[higgs,parakeet]`
  installed editable

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
  pronunciation lexicon). A test enforces this.
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
- Verification policy is a cascade: fast recogniser on every chunk, second
  opinion only on rejection, accept if either confirms. Order affects cost,
  never outcomes.

## Layout

`narrator/`: `chunking` (sentence boundary lives here — one home, three
importers), `synth` (retry ladder, cheap checks, pronunciation + acronym
spelling), `verify` (coverage scoring, folds, hard-fail rules, verifiers),
`render` (orchestration, quarantine), `audio` (DSP/mastering), `asr`
(recognisers), `backends/` (TTS engines + deterministic fake), `types`
(protocols), `cli` (the `narrate` entry point), `__init__` (the public API —
new exports go in `__all__`). `bench/` is measurement tooling; its results
justify the design.
