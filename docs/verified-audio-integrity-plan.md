# Verified audio integrity: implementation handoff

Status: plan only; implementation has not started in this handoff.

## Goal and scope

Make the audio that ships the audio whose content was verified, and prevent
verification from accepting missing repeated words or quantities assigned to the
wrong surrounding text. Balance fidelity with simplicity: repair ownership and
matching invariants rather than adding more recovery stages or quality heuristics.

The user selected priorities 1 and 2 from the project analysis. Implement these
as three independently reviewable steps, in order:

1. Prepare a take once, verify that prepared audio, and preserve it through assembly.
2. Require distinct transcript evidence for every word-boundary rescue.
3. Preserve the association between comparable numeric values and surrounding words.

Two smaller suggestions are recorded at the end as optional follow-ups; they are
not prerequisites or assumed additions to the selected scope. No implementation,
push, or merge has been performed by the planning agent.

## Starting state and ownership

- Handoff branch: `fix/verified-audio-integrity`.
- Created from local `simplify-audit`, whose HEAD at inspection was
  `635eaf5` (`Trailing dead air the gate never measured, and the Czech copula (#45)`).
- The working tree already contained staged work in `.gitignore`, `bench/`,
  `docs/`, `narrator/verify.py`, `narrator/backends/fake.py`, and several tests.
  Those changes belong to the existing work; the handoff commit includes only
  this plan. Preserve their staging and contents. Inspect `git status` and both
  staged and unstaged diffs before starting; do not reset, sweep them into an
  implementation commit, or assume they are on another checkout.
- Findings below were checked against the local working tree, not solely HEAD.
  A fresh checkout of this branch gets only committed content. Reproduce findings
  on the actual implementation baseline, and coordinate the existing staged work
  before moving implementation into a separate checkout.
- Symbols below are more stable references than line numbers after that work lands.

Read `AGENTS.md` before implementation. There is no permission to weaken the
fail-closed render contract. This is a reviewable implementation plan, not a claim
that the required independent implementation review has happened.

## Constraints shared by every step

- Verify against caller-original text after the existing explicit non-speech
  resolution; pronunciation changes remain synthesis-only.
- Preserve quarantine, the current acceptance threshold, accept-if-any cascade
  semantics, and construction of ASR at the backend's settled sample rate.
- Do not infer speaker gain, add automatic chunk loudness matching, or change
  explicit `Gap` durations. Preserve quiet delivery.
- Preserve measured comments and update their explanation when behavior changes.
- Every verifier-policy change needs tests anchored to the real transcript that
  motivated it. Synthetic mutations demonstrate a boundary; label them as such,
  and do not pass them off as newly observed audio failures.
- Cache policy must remain fail-closed. At inspection `synth.SEMANTICS == 2` and
  `verify.SEMANTICS == 8`; inspect their current values before bumping them.
  Change the relevant semantics version whenever take preparation or acceptance
  changes. Configuration keys alone do not invalidate code-policy changes.
- Keep public dataclass compatibility: do not insert fields that rebind existing
  positional constructors. Add public API only if a demonstrated need requires it.

## Step 1: one owner for prepared take audio

### Evidence and files

`narrator/synth.py::_best_attempt` measures silence on a trimmed copy but passes
raw audio to `verifier.verify`. `_sentence_split` trims verified sentences for
assembly. `narrator/render.py::render` trims each returned chunk again, including
sentence-split assemblies. `narrator/audio.py::trim_silence` uses a threshold
relative to the loudest frame, so a quiet spoken edge can be deleted after ASR
credited it. The assembly-level trim can also use a louder reference level than
the per-sentence trim.

Read-only synthetic reproduction, at 24 kHz: concatenate a one-second 220 Hz
sine of amplitude 0.001 with a one-second sine of amplitude 0.5. `trim_silence`
reduces 2.00 seconds to 1.05 seconds. This proves the deletion mechanism; it does
not establish incidence or intelligibility in real speech. Existing measurement
notes in `types.RenderReport.unscripted_silence_s` describe quiet speech lost by
a previously rejected removal policy, but do not substitute for a regression
fixture for the present change.

Primary scope: `narrator/synth.py`, `narrator/render.py`, `narrator/audio.py`,
`narrator/types.py`, `narrator/takes.py` only as needed, and their existing tests.

### Proposed design

1. Preserve raw duration/frame-cap evidence for runaway detection.
2. Prepare the candidate boundaries inside synthesis, once. Decide explicitly
   whether edge declicking belongs in the prepared buffer; avoid scattering
   content-affecting operations between synthesis and rendering.
3. Run silence checks and content verification against the prepared candidate.
   Return and cache that same candidate after it passes.
4. Sentence recovery prepares and verifies each constituent once. Preserve those
   boundaries when joining them with the existing configured sentence gap;
   retain the assembly silence check. Do not globally re-trim the assembly.
5. Rendering owns declared gain, explicit gaps, assembly, and mastering. It must
   not remove speech-bearing material from a verified take.
6. Keep `duration_s`, `shipped_s`, offsets, cached telemetry, and public return
   values truthful. If audio representation changes, update its contract and
   serialization deliberately rather than silently changing field meaning.

This does not require a second whole-episode ASR pass, a new VAD dependency, or
a new silence threshold. It closes the post-verification deletion path; it does
not claim ASR perfectly detects every loss or that mastering is sample-identical.

### Proof expected

- A recorded quiet opening/ending fixture demonstrates that passing ASR cannot
  credit speech subsequently removed by assembly. Use appropriately permitted
  audio; do not commit private reference recordings from `bench/.voices/`.
- Deterministic pipeline tests track the actual buffer presented to the verifier
  and delivered to assembly, for direct success, retry, sentence recovery, and
  cache reuse. Test observable content preservation, not helper call counts.
- Preserve existing severe trailing/interior dead-air rejection and sentence-join
  regressions. Recheck quiet final sentences adjacent to louder sentences.
- Exact gaps and report offsets remain correct, including a backend whose rate
  settles on first synthesis. Cached and uncached preparation agree.
- Bump synthesis semantics; update related docs/comments and cache tests.

## Step 2: consume distinct evidence for boundary rescue

### Evidence and files

`narrator/verify.py::coverage_detail` marks each unmatched reference word covered
when it occurs anywhere in the concatenation of unclaimed hypothesis words.
Repeated reference words can reuse one hypothesis occurrence. The `consumed`
bookkeeping currently affects diagnostics rather than the score, and can fall
back to searching an occurrence that was already used.

Reproduced on the working tree:

```text
Reference:  The coworkers help other coworkers every day.
Transcript: The co workers help other every day.
Result:     score=1.0, no word diagnostics
```

The real motivating ASR boundary disagreement is already pinned in
`tests/test_verify.py::test_hyphenated_compound_is_not_a_drop`: correct speech
containing `coworkers` was transcribed as `co-worker's`, previously scoring 0.83.
Use that measured case as the anchor; the dropped repeated occurrence above is
an adversarial mutation, not a recording claim.

Primary scope: `narrator/verify.py`, `tests/test_verify.py`, and relevant cache and
preflight regression tests.

### Proposed design and proof expected

- Allocate a distinct contiguous hypothesis character span to each successful
  rescue, respecting actual hypothesis positions and boundaries of unmatched
  regions. Concatenating separate unclaimed regions must not invent adjacency.
- Previously matched or rescued evidence cannot certify another occurrence.
  Preserve legitimate splitting/merging across adjacent transcript tokens.
- Make score and diagnostics agree about the allocation; avoid a second matching
  policy used only for diagnostic messages.
- Keep the measured `co-worker's` case passing. Reject the dropped-repeat
  mutation and dropped-sentence variants. Accept genuinely repeated split forms
  when enough evidence exists. Include overlapping/nested candidate spans and
  cases separated by already-matched words.
- Preserve the existing short-sentence and Czech merge regressions. Do not
  compensate for stricter allocation by lowering the acceptance threshold.
- Bump verifier semantics and prove takes certified under the previous policy
  cannot be reused as if certified by the new one.

## Step 3: associate comparable numerals with their context

### Evidence and files

`coverage_detail` removes numeral tokens from ordinary content alignment, then
compares sorted chunk-wide numeral collections. Equal values in different roles
therefore pass. Reproduced with score 1.0:

```text
Reference:  The account has four credits and the password has five digits.
Transcript: The account has five credits and the password has four digits.
```

A Czech counterexample also passes:

```text
Reference:  Částku dvacet korun pošlete prvnímu a třicet korun druhému.
Transcript: Částku třicet korun pošlete prvnímu a dvacet korun druhému.
```

These are transcript-level probes. First locate or collect real ASR/audio cases
that establish the numeric spelling and alignment behavior the repair must
preserve. Do not add an uncalibrated numeric grammar on the strength of these
synthetic examples alone.

Primary scope: numeral span extraction and alignment in `narrator/verify.py`,
`tests/test_verify.py`, `tests/test_preflight.py`, and take invalidation tests.
Touch `narrator/cs_numerals.py` only if actual evidence requires language-data
changes; run its documented independent audit if it changes.

### Design decision and proof expected

- Explore retaining canonical comparable values as spans associated with aligned
  neighboring content, instead of discarding position then checking a multiset.
  Choose the smallest design that handles measured ASR boundary differences.
- Merely checking a multiset per sentence is insufficient: both examples swap
  values inside a single sentence. Merely preserving numeric order also does not
  establish their attachment to context when text itself is rearranged.
- Keep digit/spelled-form equivalence, supported Czech composition, repeated
  equal values, numeral-only behavior, and measured welded-number rescues.
  Preserve established refusal behavior for uncheckable sentences and do not
  invent values for ambiguous readings. Any widening/tightening of existing
  compound-number policy is a separate, explicitly justified decision.
- Reject within-sentence and across-sentence role swaps, omissions, and added
  comparable numbers while keeping the actual recorded positive fixtures green.
- Validate direct verification, identity-transcript preflight, and both directions
  of supported spelling equivalence. Bump verifier semantics appropriately for
  the implementation sequence; do not freeze a version number from this plan.

## Optional small follow-ups, separate scope

1. **Lossless sentence punctuation.** `chunking._SENTENCE_END` consumes English
   closing quotes/brackets and omits Czech closing `“`. For example,
   `He said "Are you ready?" Then we started.` loses the closing quote, while
   `Řekl: „Jsi připravený?“ Potom jsme vyšli.` remains one scored sentence.
   Correct the one shared splitter and pin punctuation preservation, Czech
   boundaries, and existing abbreviation behavior. Assess both synthesis and
   verifier cache semantics because all importers share this boundary.
2. **Acoustic budget after pronunciation expansion.** `_best_attempt` budgets
   before `resolve_spoken`. With `spell_acronyms=True`,
   `Compare HTTP HTTPS TCP UDP and DNS.` expands from 7 to 20 spoken words while
   retaining a 6.48 s cap and 6.75 s ceiling. Budget from spoken workload excluding
   declared non-speech atoms; keep original-text verification. Calibrate on real
   expanded speech and preserve cap/short-utterance regressions. This can wait
   if those expansions are not used.

## Validation, review, and landing

Run these before each commit, and after review fixes:

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check narrator/ tests/ bench/
```

The analysis run had 401 passed, 3 skipped, and clean lint. Treat the new run as
authoritative: other staged changes existed and test counts can change. Fake
backend tests prove orchestration; they do not establish perceptual quality or
the real false-accept rate. Label evidence accordingly.

Before pushing implementation, follow `AGENTS.md` and `.claude/review.toml`:
use `/local-review` when available and satisfy its gate, or the required frontier
review otherwise. For GPT-authored implementation, frontier review must use a
capable non-GPT model. A same-family exploratory subagent is not that review.
If the required reviewer is unavailable, state the unmet requirement explicitly.
Give the reviewer exact paths, goals/non-goals, and request file:line evidence;
verify every finding before acting and name unresolved findings.

Use separate reviewable commits or PRs for the three core steps. Verifier
semantics and audio representation changes go through a branch and PR with CI
on both platforms and human review. Do not push or merge solely because the
tests pass. Keep unrelated staged work out of these commits.

Completion means: the three demonstrated integrity problems are repaired with
measured regressions, cache invalidation is correct, suite/lint and required
reviews pass, and the PR describes the actual final behavior and evidence.
No new prosody selectors, inferred gain, heavier mastering, rolling context,
acceptance-threshold relaxation, or broad benchmark rewrite is part of this work.
