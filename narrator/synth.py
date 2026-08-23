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

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from narrator import prosody
from narrator.audio import SILENCE_DROP_DB, longest_silent_run, trim_silence
from narrator.chunking import split_sentences
from narrator.takes import TakeStore, take_key
from narrator.types import Audio, Backend, ChunkResult, Verdict, Verifier, Voice

SEMANTICS = 2
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

    non_speech: tuple[str, ...] = ()
    """Exact spans that are NOT speech, declared by the caller.

    Higgs v3 ships non-speech control tokens — 84 added tokens, `<|emotion:*|>`,
    `<|sfx:*|>`, `<|style:*|>`. Each is one token, the model acts on it, and it
    is never spoken. But it is also part of the text the round-trip compares
    against the transcript, so every tagged chunk failed by construction:
    measured on an 8.5-minute episode, 73 chunks, 11 tagged, exactly those 11
    failed with the audio fine (issue #17). Forcing `quarantine=False` to use
    them switches off the primary guard for 15% of the render.

    LITERAL atoms, not a pattern. A regex was tried and abandoned: two patterns
    agreeing on a chunk's whole reference disagreed on its sentences, so a take
    stored under one was served under the other — and a shape like
    `<|[a-z_]+:[a-z_]+|>`-style shapes grant blanket "not spoken" status to every
    syntactically-shaped typo and every future token. A literal cannot bless
    what the caller did not name.

    Narrator learns no engine syntax: the caller names the spans, exactly as it
    resolves its own markup into segments elsewhere."""

    max_silence_s: float = 4.0
    """Longest interior silence a chunk may contain before it is a defect.

    Renders emit multi-second holes the script never asked for, and every other
    gate passes them: the words are all there, so coverage stays 1.000, and the
    duration ceiling is far too loose to notice — a 20-word chunk permits 16.5 s
    against ~8 s of real speech, so an 8 s hole fits inside the budget. One
    episode shipped 18.5 s of dead air reporting `failed=0`, `min coverage 1.0`
    (issue #18). Measured at 10 events over ~389 chunks.

    4.0 s refuses only the SEVERE class actually measured (7.32, 8.08, 10.44,
    10.98 s). The 1.0-1.6 s band is left to `ChunkResult.silence_s` as telemetry
    rather than refused, because a pause the script spells — `verify.py` already
    treats a punctuation-only `"..."` as exactly that — plausibly lives there,
    and refusing a class nobody has calibrated is how a gate starts rejecting
    correct audio."""

    silence_drop_db: float = SILENCE_DROP_DB
    """How far under its own speech level a stretch counts as silent. Relative,
    never absolute — see `audio.SILENCE_DROP_DB` for the counterexample."""

    def __post_init__(self) -> None:
        for atom in self.non_speech:
            if not atom.strip():
                raise ValueError("a non_speech atom cannot be empty")
            if any(c.isspace() for c in atom):
                # Whitespace-free is what makes removal commute with sentence
                # splitting: a boundary is a terminator plus whitespace, so an
                # atom can never span one, and stripping the whole chunk gives
                # the same words as stripping each sentence.
                raise ValueError(
                    f"non_speech atom {atom!r} contains whitespace; removal must "
                    "commute with sentence splitting, which whitespace breaks"
                )
            if any(c in ".!?" for c in atom):
                # An atom carrying a terminator deletes a sentence boundary from
                # the reference, so `coverage_detail` scores two sentences as
                # one — eroding the per-sentence granularity that exists to stop
                # a dropped sentence hiding in an aggregate.
                raise ValueError(
                    f"non_speech atom {atom!r} contains a sentence terminator; "
                    "that would merge two scored sentences into one"
                )
        # A NaN compares False against every bound, so the gate would be off
        # while the config still claimed to have one — the silent-disable this
        # library refuses everywhere else.
        if not math.isfinite(self.silence_drop_db) or self.silence_drop_db <= 0:
            raise ValueError(
                f"silence_drop_db={self.silence_drop_db} must be a positive, finite "
                "number of decibels; anything else silently disables the check."
            )
        if not math.isfinite(self.max_silence_s) or self.max_silence_s <= 0:
            raise ValueError(f"max_silence_s={self.max_silence_s} must be positive and finite.")
        # A sentence-split join is real digital silence (`_sentence_split`
        # allocates zeros from this), so a gap at or above the gate threshold
        # would manufacture the hole the gate then refuses.
        if self.sentence_gap_s >= self.max_silence_s:
            raise ValueError(
                f"sentence_gap_s={self.sentence_gap_s} is not below "
                f"max_silence_s={self.max_silence_s}: the rescue path would insert "
                "a silence long enough to be refused as a defect."
            )


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
    silence_s: float = 0.0
    """Longest interior silence measured on the audio that would ship."""
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
    def speech(part: str) -> str:
        spoken = apply_pronunciation(part, cfg.pronunciation) if cfg.pronunciation else part
        return spell_acronyms(spoken, lang) if cfg.spell_acronyms else spoken

    if not cfg.non_speech:
        return speech(text)
    # Atoms are OPAQUE to both transforms. Measured: a lexicon pair ("joy",
    # "radost") rewrote `<|emotion:joy|>` into `<|emotion:radost|>` — the token
    # NAME — handing the engine a control token that does not exist, silently,
    # because the tag never reaches verification either way. `spell_acronyms`
    # would do the same to an upper-case atom. Declared-not-speech means not
    # compared, not counted, and not respelled.
    return "".join(part if is_atom else speech(part)
                   for part, is_atom in _outside_atoms(text, cfg))


def resolve_reference(text: str, cfg: SynthConfig) -> str:
    """What verification compares the audio against, for this text.

    Mirror of `resolve_spoken`: one home, two callers — the ladder verifies with
    it, and the take store keys on it. Where the lexicon changes what is SPOKEN,
    this changes what is COMPARED, and neither may weaken the round-trip.

    Each atom becomes a SPACE, never the empty string. `a<atom>b` would
    otherwise fuse into `ab` and invent a word — and the space is also what
    keeps a whitespace join at every sentence boundary, which is what makes
    "an atom can never span one" true after a removal as well as before it.

    Returns `text` unchanged when no atom occurs, so declaring an atom cannot
    invalidate a stored take for a chunk that does not contain it.
    """
    if not cfg.non_speech:
        return text
    stripped = text
    for atom in cfg.non_speech:
        stripped = stripped.replace(atom, " ")
    return " ".join(stripped.split()) if stripped != text else text


def _outside_atoms(text: str, cfg: SynthConfig) -> list[tuple[str, bool]]:
    """`text` split into (span, is_atom) pieces, in order."""
    pieces: list[tuple[str, bool]] = [(text, False)]
    for atom in cfg.non_speech:
        out: list[tuple[str, bool]] = []
        for span, is_atom in pieces:
            if is_atom or atom not in span:
                out.append((span, is_atom))
                continue
            parts = span.split(atom)
            for i, part in enumerate(parts):
                if i:
                    out.append((atom, True))
                if part:
                    out.append((part, False))
        pieces = out
    return pieces


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
    keyed_rate = backend.sample_rate
    if key is not None and reuse:
        cached = store.get(key, keyed_rate, index, text)
        if cached is not None and _usable_under(cached, cfg):
            store.used()
            return cached

    result = _synthesize(text, index, backend, verifier, voice, cfg, intent)

    # Three things must still hold to file it. The split fallback may have
    # applied rise selection to a sentence since the key was computed; and a
    # backend that only settles its true rate during its first synthesis (a
    # Supertonic-style one) has just invalidated the identity the key was built
    # from, so storing would leave an entry no lookup can ever read.
    if (key is not None and result.ok and intent.cacheable
            and backend.sample_rate == keyed_rate):
        store.put(key, result, keyed_rate)
    return result


def _usable_under(cached: ChunkResult, cfg: SynthConfig) -> bool:
    """Whether a stored take is answerable by the CURRENT prosody policy.

    `wants_rise` is a caller callable and cannot be in the key, so a policy
    change is invisible to it. The chunk-level case is still covered — a chunk
    that wants a rise is never looked up at all — but a take assembled by the
    sentence-split fallback was made one sentence at a time, and a policy that
    now wants a rise for one of those sentences would never get to say so.

    Those takes are the failure path and therefore rare, so refusing to reuse
    them whenever a policy exists costs almost nothing, and asking the policy
    here instead would spend a stateful one's answers on a chunk that is not
    being rendered.
    """
    return not (cfg.wants_rise is not None and cached.recovered_by == "sentence-split")


def _take_key(
    text: str, backend: Backend, verifier: Verifier, voice: Voice, cfg: SynthConfig
) -> str | None:
    """This chunk's address in the store, or None when it cannot be addressed."""
    return take_key(
        # The slot is documented as "what verification compared the result
        # against", so it carries the reference. `spoken` covers the synthesis
        # side, which still contains the atoms, so the pair identifies both.
        text=resolve_reference(text, cfg),
        spoken=resolve_spoken(text, voice.lang, cfg),
        voice=voice,
        backend=backend,
        verifier=verifier,
        cfg=cfg,
        semantics=SEMANTICS,
    )


@dataclass
class _RiseIntent:
    """Whether rise selection touched this chunk — asked when the ladder needs it.

    The caller's policy is a caller's function and this code cannot assume it is
    pure ("rise on the first question of a paragraph" is reasonable and is
    stateful). So it is asked exactly where the previous, cacheless code asked
    it: once for the chunk, and once per sentence only if the split fallback
    actually runs. Nothing is asked speculatively for the store's benefit.

    The store then reads a FACT rather than a prediction — `rose` records that
    selection applied to something actually rendered — which is what lets the
    lookup happen before synthesis and the decision to file happen after it.
    """

    chunk: bool
    trusted: bool
    """The policy answered without raising. It never destabilises a render — an
    exception reads as "no preference" for synthesis, exactly as before — but a
    policy that cannot be asked cannot be keyed on either."""
    rose: bool = False
    """Rise selection was applied to audio this chunk shipped."""

    def __post_init__(self) -> None:
        self.rose = self.rose or self.chunk

    @property
    def cacheable(self) -> bool:
        """Rise-touched chunks are not stored, and a resolved boolean is not
        enough to key on instead.

        The contour analysis picks WHICH verified take ships, and it is
        environment-dependent — `prosody.rise_delta_checker` returns None when
        librosa is absent — so one key could serve a first verified take where
        this machine would have kept searching for a rising one. Rises are a
        minority (yes/no questions), so the carve-out is cheap and fails closed.
        """
        return self.trusted and not self.rose

    def wants(self, sentence: str, voice: Voice, cfg: SynthConfig) -> bool:
        """Ask about one sentence the split fallback is about to render alone."""
        if cfg.wants_rise is None:
            return False
        try:
            answer = bool(cfg.wants_rise(sentence, voice.lang))
        except Exception:
            self.trusted = False
            return False
        self.rose = self.rose or answer
        return answer


def resolve_rise_intent(text: str, voice: Voice, cfg: SynthConfig) -> _RiseIntent:
    """Ask the caller's intent policy about the chunk as a whole.

    Total: a policy that raises reads as "no intent" — prosody must never be able
    to fail (or even destabilise) a render — and marks the answer untrusted, so
    nothing is stored on the strength of it.
    """
    if cfg.wants_rise is None:
        return _RiseIntent(False, True)
    try:
        # Asked about SPEECH. `prosody.yes_no_question` requires `?` at
        # end-of-string and scans \w+ for wh-words, so a trailing atom defeats
        # the match outright and an atom's internals enter the word scan.
        return _RiseIntent(bool(cfg.wants_rise(resolve_reference(text, cfg),
                                               voice.lang)), True)
    except Exception:
        return _RiseIntent(False, False)


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
        # The assembly is checked as a whole, not trusted because its parts
        # passed. Two sentences can each clear the gate and still meet across a
        # join: `trim_silence` is peak-relative, so low-level residue survives it
        # while counting as silence here. A review reproduced exactly that — an
        # assembly measuring 9.12 s of interior silence returned ok=True,
        # recovered_by="sentence-split", and was stored for reuse.
        if audio_ is not None and longest_silent_run(
            audio_, backend.sample_rate, cfg.silence_drop_db
        ) > cfg.max_silence_s:
            audio_ = None
        if audio_ is not None:
            return ChunkResult(
                index=index, text=text, audio=audio_,
                duration_s=len(audio_) / backend.sample_rate,
                attempts=cfg.max_attempts + split_spent,
                ok=True, coverage=coverage_, recovered_by="sentence-split",
                # Measured on the ASSEMBLY, not inherited from a sentence: the
                # joins are part of what ships. `SynthConfig.__post_init__`
                # keeps `sentence_gap_s` below the threshold, so a join can
                # never be mistaken for the defect.
                silence_s=longest_silent_run(audio_, backend.sample_rate,
                                             cfg.silence_drop_db),
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
            diagnostic = verifier.verify(attempt.audio, resolve_reference(text, cfg),
                                         voice.lang)
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
    reference = resolve_reference(text, cfg)
    # Counted from the REFERENCE, because a declared atom is not a spoken word.
    # Two real words carrying three atoms read as five, and the duration FLOOR
    # then demands 1.11 s of audio for ~0.8 s of correct speech — a false
    # failure. The ceiling moves too, but only marginally; the floor is why.
    words = len(reference.split())
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
        # Measured on the TRIMMED audio, which is what render actually ships
        # (render.py applies trim_silence before stitching). Measuring the raw
        # buffer would fail chunks for leading or trailing silence that is about
        # to be removed, and `longest_silent_run` only counts interior runs for
        # the same reason.
        silence_s = longest_silent_run(
            trim_silence(audio, backend.sample_rate), backend.sample_rate,
            cfg.silence_drop_db,
        )
        silence_ok = silence_s <= cfg.max_silence_s
        # Only pay for verification when the cheap checks already passed. The
        # silence check belongs with them: it is arithmetic over one buffer,
        # where verification is an ASR call, and coverage is structurally blind
        # to this defect anyway — silence between words contains no words.
        verdict = (
            verifier.verify(audio, reference, voice.lang)
            if duration_ok and silence_ok and not hit_cap
            else Verdict(False, 0.0)
        )
        attempt = _Attempt(audio, duration, duration_ok, verdict, hit_cap,
                           silence_s=silence_s,
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


def _coalesce_atom_only(sentences: list[str], cfg: SynthConfig) -> list[str]:
    """Fold sentences that are ONLY declared atoms into a neighbour.

    `split_sentences("Je to tak? <|emotion:joy|>")` yields a trailing sentence
    with no speech in it. Rendered alone it can never verify, so the rescue
    aborts and a tagged chunk loses the recovery path that exists because the
    whole-chunk failure it rescues is stochastic — 11 of 73 chunks were tagged
    in the motivating episode.

    Mirrors `_merge_orphans` in chunking, for the same measured reason: short
    inputs are where engines are least stable. Direction is fixed rather than
    guessed — a leading run folds FORWARD, anything else folds BACK — because an
    interior atom has no universally correct neighbour and a coin-flip there
    would move the atom's effect to the wrong side of a boundary.
    """
    if not cfg.non_speech:
        return sentences

    def speechless(sentence: str) -> bool:
        return not resolve_reference(sentence, cfg).strip()

    out: list[str] = []
    for sentence in sentences:
        if speechless(sentence) and out:
            out[-1] = f"{out[-1]} {sentence}"
        else:
            out.append(sentence)
    # A leading run had no predecessor to fold into; give it to what follows.
    while len(out) > 1 and speechless(out[0]):
        out[1] = f"{out[0]} {out[1]}"
        out.pop(0)
    return out


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
    sentences = _coalesce_atom_only(split_sentences(text), cfg)
    if len(sentences) < 2:
        return None, 0.0, 0

    pieces: list[Audio] = []
    gap: Audio | None = None
    worst = 1.0
    attempts = 0
    for sentence in sentences:
        # Asked here, where the sentence is actually about to be rendered alone.
        # Asking up front for the store's benefit would consume a stateful
        # policy's answers for sentences no ladder ever reaches.
        attempt = _best_attempt(sentence, backend, verifier, voice, cfg,
                                intent.wants(sentence, voice, cfg))
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
        # Trimmed before assembly, so the audio that ships IS the audio the
        # checks measured. Joining the raw buffer let a sentence carry a long
        # trailing pad through: the silence check measured a trimmed copy and
        # passed, then the pad was embedded inside the assembled chunk, reported
        # clean, and stored for reuse. `render`'s own trim only reaches the
        # assembled chunk's outer edges, so it could never remove an interior
        # one.
        pieces.extend([trim_silence(attempt.audio, backend.sample_rate), gap])
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
        silence_s=attempt.silence_s,
    )
