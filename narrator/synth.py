"""Synthesize one chunk, and keep trying until it is right or provably isn't.

The escalation ladder, in order:

1. generate, bounded by a frame cap
2. cheap checks: did it hit the cap, is the duration plausible
3. the real check: does the audio say the words
4. retry — the failure is stochastic, so a re-roll genuinely helps
5. split into sentences and render each alone, where "dropped a whole sentence"
   stops being expressible
6. give up loudly

Step 6 matters as much as the rest. A chunk that fails every path must not reach
the assembly stage: shipping it is exactly the silent corruption this library
exists to prevent, and "best effort" here means "plausible audio saying the wrong
thing, indistinguishable from success".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from narrator import prosody
from narrator.chunking import split_sentences
from narrator.takes import TakeStore, take_key
from narrator.types import Audio, Backend, ChunkResult, Verdict, Verifier, Voice

SEMANTICS = 1
"""Version of the ladder's own behaviour, for the take store's key.

Bump it whenever a change here alters which take ships or how one is made —
ranking, attempt budget, the split fallback, the cheap checks. The config fields
are keyed on automatically; this covers the code around them, so that a fix
re-renders rather than certifying old audio as if this version had produced it.
"""


@dataclass(frozen=True)
class SynthConfig:
    temperature: float = 0.4
    """Measured: acronym rendering degrades monotonically above this. At 0.4 a
    real engine produced "SHA-256" twice; at 1.0 it produced "ŠAA256" and
    "SHH256"."""

    max_attempts: int = 3
    """Best-of-N with ASR selection drove hard failures 0.269 -> 0.038 at N=3."""

    words_per_second: float = 2.5
    frame_headroom: float = 1.6
    frame_headroom_s: float = 2.0
    """ABSOLUTE headroom, added on top of the multiplier.

    A pure multiplier truncates short utterances by construction: "Jen firmu."
    is two words, so 0.8 s expected x 1.6 = 1.28 s — less than the real leading
    and trailing silence. That capped every short sentence, which broke the
    sentence-split fallback for every chunk containing one, which is precisely
    the chunks the fallback exists to rescue."""

    min_frame_cap_s: float = 4.0
    duration_floor_wps: float = 4.5
    duration_ceiling_base_s: float = 1.5
    duration_ceiling_per_word_s: float = 0.75
    sentence_gap_s: float = 0.12
    allow_sentence_split: bool = True

    spell_acronyms: bool = False
    """Read all-caps tokens as letter names for the render language. Off by
    default: it changes how the narration sounds, which is the caller's call."""

    wants_rise: Callable[[str, str], bool] | None = None
    """(text, lang) -> should this chunk end in a rising contour?

    None — the default — disables prosody selection entirely; behavior is
    identical to a build without this feature. Off by default for the same
    reason spell_acronyms is: it changes how the narration sounds. Intent
    must come from the caller because punctuation cannot supply it —
    wh-questions end in `?` and measurably go DOWN (31/32 verified takes,
    bench/RESULTS.md §11/§11.7), so a `?` trigger would push rises onto the
    one category that must not have them. `narrator.prosody.yes_no_question` is the offered
    default policy."""

    rise_threshold_st: float = prosody.RISE_THRESHOLD_ST
    """Semitone delta a verified take must reach to stop the ladder early.
    Below it the ladder keeps generating; if nothing clears it, the FIRST
    verified take ships — prosody is a preference, never a gate."""

    pronunciation: tuple[tuple[str, str], ...] = ()
    """Written form -> spoken form, applied ONLY at synthesis.

    A pronunciation lexicon and a round-trip verifier are in direct conflict if
    the substitution happens upstream: the engine is asked to say "Kalleh" so the
    name is not collapsed to "Cal", the ASR faithfully reports "Kalle", and the
    verifier — comparing against the respelled text — calls correct audio a
    failure. Measured on a real render: 0.88 against the substituted form, 1.00
    against the original.

    So narrator applies it here, immediately before the backend call, and always
    verifies against the caller's original text. Chunking also runs on the
    original, so chunk boundaries do not shift when the lexicon changes."""


def frame_cap(words: int, fps: int, cfg: SynthConfig) -> int:
    expected = words / cfg.words_per_second
    seconds = max(expected * cfg.frame_headroom + cfg.frame_headroom_s, cfg.min_frame_cap_s)
    return int(seconds * fps)


def duration_bounds(words: int, cfg: SynthConfig) -> tuple[float, float]:
    floor = words / cfg.duration_floor_wps
    ceiling = cfg.duration_ceiling_base_s + cfg.duration_ceiling_per_word_s * words
    return floor, ceiling


@dataclass
class _Attempt:
    audio: Audio
    duration: float
    duration_ok: bool
    verdict: Verdict
    hit_cap: bool = False
    prior_failures: int = 0
    """Failed or raised attempts before success was first secured.
    `recovered_by="retry"` keys off this: with rise selection a later take
    can ship after a clean first verification, and failures burned during
    that optional search are not recoveries — the provenance is pinned to
    the FIRST verified take (frontier review caught the misreport)."""
    calls_spent: int = 0
    """Generations actually spent by the ladder that returned this attempt.
    Rise selection can spend more calls than the shipped take's ordinal,
    and ChunkResult.attempts must report real cost."""

    @property
    def ok(self) -> bool:
        return self.duration_ok and not self.hit_cap and self.verdict.ok

    @property
    def rank(self) -> tuple[int, float]:
        """How good a *failed* attempt is, for picking the least-bad to report.

        Duration-valid outranks duration-invalid, then coverage. The predecessor
        ranked on coverage alone and gave a skipped check a perfect 1.0, so an
        attempt that failed on duration could never be beaten — a later attempt
        that passed everything was generated, then discarded.
        """
        return (1 if (self.duration_ok and not self.hit_cap) else 0, self.verdict.coverage)


def apply_pronunciation(text: str, pairs: tuple[tuple[str, str], ...]) -> str:
    """Longest key first, so a longer entry is not clipped by a shorter prefix."""
    for written, spoken in sorted(pairs, key=lambda kv: len(kv[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(written)}\b", spoken, text)
    return text


# Letter names per language — general linguistic data, like the numeral tables
# in verify.py. Spelled phonetically so any engine can say them; the ASR side
# needs nothing, because recognisers normalize spelled letters back into the
# acronym (measured: a spelled six-letter pair came back verbatim as the two
# acronyms, coverage 1.00).
_LETTER_NAMES = {
    "en": {
        "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "eff",
        "g": "gee", "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "ell",
        "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "are",
        "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "double you",
        "x": "ex", "y": "why", "z": "zed",
    },
    "cs": {
        "a": "á", "b": "bé", "c": "cé", "d": "dé", "e": "é", "f": "ef",
        "g": "gé", "h": "há", "i": "í", "j": "jé", "k": "ká", "l": "el",
        "m": "em", "n": "en", "o": "ó", "p": "pé", "q": "kvé", "r": "er",
        "s": "es", "t": "té", "u": "ú", "v": "vé", "w": "dvojité vé",
        "x": "iks", "y": "ypsilon", "z": "zet",
    },
}

_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")


def resolve_spoken(text: str, lang: str, cfg: SynthConfig) -> str:
    """What the engine is actually asked to say, for this text under this config.

    One home, two callers: the ladder speaks it, and the take store keys on it.
    Keying on the resolved form rather than on (text + the whole lexicon) is what
    keeps a pronunciation entry local — adding a respelling for a name that
    appears in three chunks leaves the other eighty-five addressable.
    """
    spoken = apply_pronunciation(text, cfg.pronunciation) if cfg.pronunciation else text
    return spell_acronyms(spoken, lang) if cfg.spell_acronyms else spoken


def spell_acronyms(text: str, lang: str) -> str:
    """Respell every all-caps token as its letter names — how acronyms are read.

    Runs AFTER the pronunciation lexicon, so a project that wants a word-like
    reading for a specific acronym overrides it with a lexicon entry; anything
    still all-caps gets the general treatment. The verifier needs no pairing:
    recognisers write spelled letters back as the acronym itself.
    """
    names = _LETTER_NAMES.get(lang.split("-")[0])
    if names is None:
        return text
    return _ACRONYM.sub(lambda m: " ".join(names[c] for c in m.group().lower()), text)


def synthesize_chunk(
    text: str,
    index: int,
    backend: Backend,
    verifier: Verifier,
    voice: Voice,
    cfg: SynthConfig = SynthConfig(),
    store: TakeStore | None = None,
    reuse: bool = True,
) -> ChunkResult:
    """Render one chunk. `ChunkResult.ok` is the only trustworthy field.

    With a `store`, a take that was already made for these exact inputs is
    returned instead of being generated again, and a new one is filed as soon as
    it verifies — so a killed render resumes and an edited script pays only for
    what changed. `reuse=False` forces a fresh generation for this chunk and
    overwrites the entry: it is how a caller asks a sampled model for another
    take of audio that verifies but does not sound right.

    The store is consulted around the ladder, never inside it. A single
    generation is stochastic and may be wrong; only the take the ladder decided
    to ship is worth keeping, and only when it passed.
    """
    intent = resolve_rise_intent(text, voice, cfg)
    key = (_take_key(text, backend, verifier, voice, cfg)
           if store is not None and intent.cacheable else None)
    if key is not None and reuse:
        cached = store.get(key, backend.sample_rate, index, text)
        if cached is not None:
            return cached

    result = _synthesize(text, index, backend, verifier, voice, cfg, intent)

    if key is not None and result.ok:
        store.put(key, result, backend.sample_rate)
    return result


def _take_key(
    text: str, backend: Backend, verifier: Verifier, voice: Voice, cfg: SynthConfig
) -> str | None:
    """This chunk's address in the store, or None when it cannot be addressed."""
    return take_key(
        text=text,
        spoken=resolve_spoken(text, voice.lang, cfg),
        voice=voice,
        backend=backend,
        verifier=verifier,
        cfg=cfg,
        semantics=SEMANTICS,
    )


@dataclass(frozen=True)
class _RiseIntent:
    """What the caller's policy said about this chunk — asked exactly once.

    Once, because the policy is a caller's function and this code cannot assume
    it is pure. It is consulted for the whole chunk AND for each sentence the
    split fallback might render alone, and both the synthesis path and the
    caching decision then read the SAME answers. Asking twice let the two
    disagree: a policy answering False for the cache check and True inside the
    ladder would store rise-selected audio under the rule that says it must not
    be stored.
    """

    chunk: bool
    sentences: tuple[bool, ...]
    trusted: bool
    """The policy answered without raising. It never destabilises a render —
    an exception reads as "no preference" for synthesis, exactly as before —
    but a policy that cannot be asked cannot be keyed on either."""

    @property
    def cacheable(self) -> bool:
        """Rise-wanting chunks are not stored, and that carve-out is why the
        answers above are resolved rather than keyed on.

        A resolved boolean would not be enough on its own: the contour analysis
        picks WHICH verified take ships, and it is environment-dependent —
        `prosody.rise_delta_checker` returns None when librosa is absent — so one
        key could serve a first verified take where this machine would have kept
        searching for a rising one. Rises are a minority (yes/no questions), so
        the carve-out is cheap and it fails closed.
        """
        return self.trusted and not (self.chunk or any(self.sentences))


def resolve_rise_intent(text: str, voice: Voice, cfg: SynthConfig) -> _RiseIntent:
    """Ask the caller's intent policy about a chunk and each of its sentences.

    Total: a policy that raises reads as "no intent" — prosody must never be able
    to fail (or even destabilise) a render — and marks the answers untrusted, so
    nothing is cached on the strength of them.

    Asked per candidate, not in one guarded block. The sentences are queried
    eagerly here but are only ever USED if the chunk fails and the split fallback
    runs, so letting one of them raise discard the chunk's own answer would drop
    a rise the ladder would have honoured before this resolution existed — a
    feature regression paid for a cache decision.
    """
    if cfg.wants_rise is None:
        return _RiseIntent(False, (), True)

    def ask(candidate: str) -> tuple[bool, bool]:
        try:
            return bool(cfg.wants_rise(candidate, voice.lang)), True
        except Exception:
            return False, False

    chunk, chunk_ok = ask(text)
    # Only when the fallback could actually render them alone. `_sentence_split`
    # gives up below two sentences, so a single-sentence chunk would otherwise be
    # asked twice about the same string — which for a STATEFUL policy (an
    # alternating one, say) consumes two answers where the ladder consumes one.
    sentences = split_sentences(text)
    answers = [ask(sentence) for sentence in sentences] if len(sentences) > 1 else []
    return _RiseIntent(
        chunk=chunk,
        sentences=tuple(answer for answer, _ in answers),
        trusted=chunk_ok and all(ok for _, ok in answers),
    )


def _synthesize(
    text: str,
    index: int,
    backend: Backend,
    verifier: Verifier,
    voice: Voice,
    cfg: SynthConfig,
    intent: _RiseIntent,
) -> ChunkResult:
    """The ladder itself, with no store in the picture."""
    attempt = _best_attempt(text, backend, verifier, voice, cfg, intent.chunk)

    if attempt is not None and attempt.ok:
        # "retry" means a failure was recovered. Keyed off prior_failures, not
        # number: rise selection can select ordinal 2 after a *verified* first
        # take, and calling that a recovery would misreport a healthy chunk.
        return _result(index, text, attempt, recovered_by="retry" if attempt.prior_failures else "")

    split_spent = 0
    if cfg.allow_sentence_split:
        audio_, coverage_, split_spent = _sentence_split(text, backend, verifier, voice, cfg,
                                                         intent)
        if audio_ is not None:
            return ChunkResult(
                index=index, text=text, audio=audio_,
                duration_s=len(audio_) / backend.sample_rate,
                attempts=cfg.max_attempts + split_spent,
                ok=True, coverage=coverage_, recovered_by="sentence-split",
            )
        # The failed split's generations were still paid for; the failed
        # result must report them, or six real calls read as three.

    if attempt is not None and not attempt.verdict.transcript and attempt.audio.size:
        # Diagnostics for the chunk we are about to report as failed. Verification
        # was skipped because a cheap check already failed, so there is no
        # transcript and no named sentence — which is exactly what a caller needs
        # to tell "it said the wrong thing" from "it stopped early". Costs one
        # extra ASR call, on failure only.
        #
        # This MUST NOT be able to rescue the attempt. Assigning the fresh verdict
        # wholesale did exactly that: `_Attempt.ok` reads `verdict.ok`, so a flaky
        # ASR returning success on this second call turned a failed chunk into a
        # passing one without generating any new audio. Copy the diagnostic text
        # across; keep the failure.
        try:
            diagnostic = verifier.verify(attempt.audio, text, voice.lang)
            attempt.verdict = replace(diagnostic, ok=False)
        except Exception:
            pass

    if attempt is None:
        # Every attempt raised. There is no audio to hand back, and inventing
        # silence here would put a hole in the episode that reads as a pause.
        return ChunkResult(
            index=index, text=text, audio=np.zeros(0, dtype=np.float32),
            duration_s=0.0, attempts=cfg.max_attempts + split_spent, ok=False,
            coverage=0.0, dropped_sentence=text,
        )
    return _result(index, text, attempt, extra_calls=split_spent)


def _rise_checker(wanted: bool) -> Callable | None:
    """The resolved contour checker when this text should rise, else None.

    `wanted` is decided by `resolve_rise_intent`, not asked again here: the
    caller's policy is consulted once per chunk and every path reads that answer.
    """
    if not wanted:
        return None
    try:
        # Resolution can fail beyond ImportError — a broken librosa install
        # raises whatever it raises at import time. That must degrade to "no
        # preference", not abort a render that synthesized fine yesterday.
        # (Tests inject a stub by monkeypatching this resolver — fake-backend
        # audio is a stamped tone no real F0 tracker should interpret.)
        return prosody.rise_delta_checker()
    except Exception:
        return None


def _best_attempt(
    text: str, backend: Backend, verifier: Verifier, voice: Voice, cfg: SynthConfig,
    wants_rise: bool = False,
) -> _Attempt | None:
    words = len(text.split())
    cap = frame_cap(words, backend.frames_per_second(), cfg)
    floor, ceiling = duration_bounds(words, cfg)
    best: _Attempt | None = None
    first_ok: _Attempt | None = None
    failures = 0
    rise_check = _rise_checker(wants_rise)

    spoken = resolve_spoken(text, voice.lang, cfg)
    # The engine is handed a voice with no level correction. `gain_db` is applied
    # by `render` AFTER synthesis, so no backend has any business reading it —
    # and stripping it here is what lets the take store leave it out of the key,
    # which is what makes "-3 dB, listen, -4 dB" cost nothing instead of 88
    # generations. Structural, not a convention: the bundled fake already keys
    # its per-voice amplitude on the Voice object itself, so a backend CAN see
    # a field it should not act on.
    engine_voice = replace(voice, gain_db=0.0) if voice.gain_db else voice

    for number in range(1, cfg.max_attempts + 1):
        try:
            audio = backend.synthesize(spoken, engine_voice,
                                       max_frames=cap, temperature=cfg.temperature)
        except Exception:
            # One bad attempt must not lose the whole render. The predecessor had
            # no guard here, so a transient error at chunk 80 discarded fifteen
            # minutes of completed work.
            failures += 1
            continue

        duration = len(audio) / backend.sample_rate
        # Reaching the cap means generation was still going when it was stopped.
        # This must be its own signal: for typical chunk lengths the cap lands
        # *below* the duration ceiling, so a runaway is truncated to cap length
        # and then sails through the ceiling check. (Not true for the very
        # shortest inputs, where the 4 s floor exceeds the ceiling — but both
        # checks are required, so that ordering never creates a pass.) Without
        # this the cap bounds the cost of a runaway without ever detecting one.
        # Plain attribute access on purpose: the protocol requires the flag, and
        # a permissive getattr default classified a non-conforming backend as
        # cap-honouring — every long output misread as a runaway, retries and
        # sentence-split burned on correct audio. Failing loudly is the
        # library's stated preference (see the sample_rate guard in render()).
        hit_cap = (
            backend.honours_frame_cap
            and duration >= (cap / backend.frames_per_second()) - 1e-6
        )
        duration_ok = floor <= duration <= ceiling
        # Only pay for verification when the cheap checks already passed.
        verdict = (
            verifier.verify(audio, text, voice.lang)
            if duration_ok and not hit_cap
            else Verdict(False, 0.0)
        )
        attempt = _Attempt(audio, duration, duration_ok, verdict, hit_cap,
                           prior_failures=failures, calls_spent=number)

        if attempt.ok:
            if rise_check is None:
                return attempt
            # F0 runs ONLY here: on a verified take of a rise-wanting chunk.
            # Failed takes and statements never pay for it.
            try:
                delta = rise_check(audio, backend.sample_rate)
            except Exception:
                delta = None
            if first_ok is not None:
                # Success was secured before this take: failures burned
                # during the optional rise search are not recoveries, and
                # reporting them as "retry" misread a healthy chunk.
                attempt.prior_failures = first_ok.prior_failures
            if delta is None or delta >= cfg.rise_threshold_st:
                # Confident rise — or unmeasurable, which ships as a COST
                # policy, not a claim about the text: measurability is
                # per-take stochastic (a real "Máš teď chvilku?" measured on
                # two takes of three), but chasing a measurable contour with
                # the remaining budget has unknown payoff, and None also
                # covers a broken analysis, where retrying buys nothing.
                return attempt
            if first_ok is None:
                first_ok = attempt
        else:
            failures += 1
            if best is None or attempt.rank > best.rank:
                best = attempt

    # No verified take cleared the rise threshold: ship the FIRST verified
    # take. Preferring the largest sub-threshold delta is plausible but
    # unmeasured (bench/RESULTS.md §11) — the conservative choice cannot be
    # worse than the pre-selection pipeline.
    chosen = first_ok if first_ok is not None else best
    if chosen is not None:
        chosen.calls_spent = cfg.max_attempts
    return chosen


def _sentence_split(
    text: str, backend: Backend, verifier: Verifier, voice: Voice, cfg: SynthConfig,
    intent: _RiseIntent,
) -> tuple[Audio | None, float, int]:
    """Render sentence by sentence. Audio is None unless every sentence passes.

    The third element is the synthesis attempts actually spent across the
    sentences — reported on FAILURE too, because the failed split's
    generations were still paid for. The old `+ len(sentences)` undercounted
    success by up to max_attempts per sentence, and the old None-on-failure
    made six real calls read as three in the failed chunk's report.

    At this granularity "the model dropped a sentence" stops being expressible:
    a sentence rendered alone either succeeds or fails visibly. It is the same
    containment a strictly per-sentence pipeline gets for free, applied only
    where it is needed so the rest keeps the prosody that chunking buys.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None, 0.0, 0

    pieces: list[Audio] = []
    gap: Audio | None = None
    worst = 1.0
    attempts = 0
    for position, sentence in enumerate(sentences):
        # The caller's policy was asked about these same sentences once, up
        # front; this reads that answer rather than asking again.
        wants = intent.sentences[position] if position < len(intent.sentences) else False
        attempt = _best_attempt(sentence, backend, verifier, voice, cfg, wants)
        if attempt is None or not attempt.ok:
            attempts += attempt.calls_spent if attempt is not None else cfg.max_attempts
            return None, 0.0, attempts
        if gap is None:
            # Allocated only after a sentence has actually been synthesized.
            # When every whole-chunk attempt raised, this fallback makes the
            # backend's first real call, and a Supertonic-style backend only
            # corrects its declared rate during that call — a gap allocated
            # before the loop used the stale 44100 and 0.12 s of join played
            # as 0.22 s at the settled rate.
            gap = np.zeros(int(cfg.sentence_gap_s * backend.sample_rate), dtype=np.float32)
        attempts += attempt.calls_spent
        pieces.extend([attempt.audio, gap])
        worst = min(worst, attempt.verdict.coverage)

    return np.concatenate(pieces[:-1]), worst, attempts


def _result(index: int, text: str, attempt: _Attempt, recovered_by: str = "",
            extra_calls: int = 0) -> ChunkResult:
    return ChunkResult(
        index=index, text=text, audio=attempt.audio, duration_s=attempt.duration,
        attempts=attempt.calls_spent + extra_calls, ok=attempt.ok,
        coverage=attempt.verdict.coverage,
        dropped_sentence=attempt.verdict.dropped_sentence,
        transcript=attempt.verdict.transcript, recovered_by=recovered_by,
        word_diagnostics=attempt.verdict.word_diagnostics,
    )
