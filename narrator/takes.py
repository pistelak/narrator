"""Persist verified takes, so an edit re-renders only what changed.

Without this, `render` touched disk exactly once — after the last chunk — and two
costs followed from the same gap. Changing six characters in a script, to dodge an
ASR boundary ambiguity, re-synthesised all 88 chunks of an 18-minute episode; 87 of
them were byte-identical in input to takes that already existed, at ~25 minutes of
wall clock per pass. And a run killed at chunk 60 of 88 left no artifact at all.
The quarantine path made it worse: a render that fails one chunk raises and discards
the other 87 — which is exactly the render about to be re-run after fixing one line.

The unit stored here is the **verified take**: the audio the retry ladder decided to
ship, not a single generation. A generation is stochastic and may be wrong, so
caching one would freeze a failure and turn the ladder into a no-op. Only `ok=True`
results are written, for the same reason in reverse: a stored failure makes a
transient failure permanent.

What goes in the key follows `_cache_key`'s rule (narrator/backends/higgs.py), one
level up: **include everything by default, exclude only with a written reason.**
Under-keying serves audio that was made for different inputs, under a report that
still says clean — the failure this library exists to prevent. Over-keying only
costs time. The same asymmetry decides every open question here: an object that
cannot state what it is disables the store rather than being guessed at.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from narrator.types import ChunkResult, Voice

FORMAT_VERSION = 2
"""Bumped when the stored layout or the key's composition changes, which
invalidates every existing entry rather than reinterpreting it."""

_EXCLUDED_SYNTH_FIELDS = frozenset({
    "pronunciation",
    "spell_acronyms",
    # Both reach the audio ONLY through the spoken form, which is in the key
    # already. Keying on them directly would throw away every take in the
    # episode when a lexicon entry is added for a word that appears in three
    # chunks.
    "wants_rise",
    # A caller callable, not data. Chunks it applies to are not stored at all —
    # see `synth._RiseIntent` for why a resolved boolean is not enough.
})
"""Everything else in SynthConfig is keyed on, INCLUDING fields added later:
the key is built by walking the dataclass, so a new knob that changes the audio
is covered the day it lands rather than the day someone remembers this file."""


def content_digest(path: Path) -> str:
    """Digest a file's contents, for a cache key that survives an in-place edit.

    Content, not size and mtime: a same-size replacement that preserves the
    timestamp — or is written inside a filesystem's timestamp granularity, which
    `st_mtime_ns` reports in nanoseconds without resolving to them — keeps the old
    key and serves the previous speaker's audio. Verification would not catch it,
    because ASR checks the words, not who said them, so the wrong voice ships with
    a clean report. Measured, because it runs once per synthesis call: 0.33 ms for
    a 10 s reference, 2.2 ms for a 60 s one — against a render measured in minutes.
    """
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def package_version(name: str) -> str | None:
    """Installed version of a dependency, or None when it cannot be established.

    In the key because the engines and recognisers are lower-bounded, not pinned
    (see pyproject): upgrading mlx-audio changes the samples a prompt produces, and
    upgrading a recogniser changes the verdict attached to them, with no other part
    of the key moving. An upgrade should re-render, not certify old audio as if the
    new stack had made it.

    None rather than a sentinel string, and the identities that read it return
    None in turn, disabling the store. A vendored or editable checkout carrying no
    distribution metadata is an unknown, and one shared spelling of "unknown"
    would make two different unknowns look like the same dependency — the same
    mistake as guessing at an object with no identity at all.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - environment-dependent
        return None


def identity_of(obj: Any) -> str | None:
    """The object's declared cache identity, or None when it has none.

    None disables the store for the whole render, read AND write. Backends,
    verifiers and recognisers are protocols this library does not own, and it
    cannot digest an arbitrary implementation — so it refuses to guess, exactly
    as it refuses to guess a speaker's level. A weaker rule here would let a
    stale take ship under a clean report, which is the one outcome no amount of
    saved wall clock pays for.

    Read with getattr, never `isinstance`: `Backend`, `Verifier` and `ASR` are
    @runtime_checkable, so declaring `identity` ON them would make it structurally
    REQUIRED, and every third-party implementation — the ones this fallback exists
    to keep working — would stop being one. See `types.Identified`.
    """
    try:
        value = getattr(obj, "identity", None)
    except Exception:
        # A property that raises is not an identity. `getattr`'s default only
        # covers AttributeError, so without this a caller's broken accessor
        # takes down a render the store was only supposed to speed up.
        return None
    return value if isinstance(value, str) and value else None


def _class_id(obj: Any) -> str:
    """Module and qualified name — what an override is, not just what it is called.

    A subclass inherits every field an identity is built from, so the class has to
    be part of it or a stricter verifier reuses what a laxer one accepted. The
    module belongs there too: two subclasses can share a name in different files
    and mean entirely different rules.
    """
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _voice_identity(voice: Voice) -> str | None:
    """The voice, as far as anything that reaches the audio is concerned.

    `gain_db` is deliberately absent, and that is safe only because the backend
    never sees it: `synth` zeroes it before the engine call, since gain is applied
    after synthesis by `render`. Keying on it would make `-3 dB -> listen -> -4 dB`
    cost 88 generations to change a multiplier.

    An unreadable reference disables the store instead of raising — a missing clip
    is the backend's error to report, in its own words, and the tests construct
    `Voice(Path("nonexistent.wav"))` on purpose.

    JSON-encoded rather than joined on a separator, because a transcript is
    arbitrary caller text: any delimiter that can appear inside a field makes two
    different voices produce one identity, and a colliding voice identity means
    audio conditioned on one speaker is served for another.
    """
    reference = ""
    if voice.audio_path is not None:
        try:
            reference = json.dumps([str(voice.audio_path.resolve()),
                                    content_digest(voice.audio_path)])
        except OSError:
            return None
    return json.dumps([reference, voice.transcript, voice.lang, voice.preset])


def take_key(
    *,
    text: str,
    spoken: str,
    voice: Voice,
    backend: Any,
    verifier: Any,
    cfg: Any,
    semantics: int,
) -> str | None:
    """The address of this chunk's take, or None when it must not be cached.

    Both texts are keyed on: `spoken` is what the engine was asked to say, `text` is
    what verification compared the result against, and a stored verdict is a claim
    about the pair. Neither the chunk index nor `max_chars` appears — the index
    would make inserting a paragraph at the top of a script re-render everything
    below it, and `max_chars` only ever reaches the audio by changing the chunk text
    that is already here.

    `semantics` is the caller's own behaviour version (`synth.SEMANTICS`), passed in
    rather than imported because `synth` imports this module. The verifier's
    equivalent rides inside its identity.
    """
    backend_id = identity_of(backend)
    verifier_id = identity_of(verifier)
    voice_id = _voice_identity(voice)
    if backend_id is None or verifier_id is None or voice_id is None:
        return None

    material = {
        "format": FORMAT_VERSION,
        "semantics": semantics,
        "text": text,
        "spoken": spoken,
        "voice": voice_id,
        "backend": backend_id,
        "verifier": verifier_id,
        "synth": {
            f.name: repr(getattr(cfg, f.name))
            for f in fields(cfg)
            if f.name not in _EXCLUDED_SYNTH_FIELDS
        },
    }
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class TakeStore:
    """A directory of verified takes, addressed by content.

    Audio is stored as `synthesize_chunk` returns it — before `trim_silence`,
    `declick`, `apply_gain` and `master`. Those are deterministic and cheap, so
    they are recomputed every run, and a level or mastering change costs nothing.
    Float WAV rather than .npy on both counts that matter: float32 round-trips
    exactly, where PCM_16 would make a reused take audibly equal but not bit-equal
    to a fresh render, and an individual take can be listened to, which is how this
    project debugs.
    """

    root: Path
    write_failures: int = field(default=0, init=False)
    """Takes that could not be written. Counted, never raised: the store is an
    optimisation, and a full disk must not fail a render that synthesised fine."""
    usable: int = field(default=0, init=False)
    """Chunks of THIS render that are addressable in the store — read from it or
    filed into it. What a caller needs after a refusal is not how many files the
    directory holds (most of which may belong to another script) but how much of
    the render they are about to repeat is already paid for."""

    def get(self, key: str, sample_rate: int, index: int, text: str) -> ChunkResult | None:
        """The stored take for `key`, or None for any reason at all.

        Every failure mode here is a miss, never an exception: a truncated write, a
        half-populated directory, a rate that does not match the backend now. The
        cost of a miss is one re-synthesis; the cost of trusting a bad entry is the
        wrong audio under a clean report.
        """
        sidecar = self.root / f"{key}.json"
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("format") != FORMAT_VERSION or meta.get("sample_rate") != sample_rate:
                return None
            if meta.get("key") != key:
                # The entry names the key it was filed under, so a take that was
                # copied — or a directory merged from another machine — cannot be
                # served for a chunk it was never made for. The filename alone is
                # a claim by whoever wrote it; this is the take's own.
                return None

            import soundfile as sf

            audio, rate = sf.read(str(self.root / f"{key}.wav"), dtype="float32")
            if rate != sample_rate or audio.ndim != 1:
                return None
            if hashlib.blake2b(audio.tobytes(), digest_size=16).hexdigest() != meta.get("digest"):
                # The sidecar is written last, so a killed run leaves an
                # uncommitted wav and no entry at all. This catches the rest:
                # a corrupted file, or a wav paired with another take's sidecar
                # while an entry is being overwritten.
                return None
            # Inside the guard, like everything above it: a sidecar that parses
            # but is missing a field is corruption too, and corruption is a miss.
            result = ChunkResult(
                index=index,
                text=text,
                audio=audio,
                # Coerced, not just fetched: a sidecar carrying "bad" where a
                # number belongs parses as JSON and matches the digest, and would
                # otherwise return a hit that crashes a progress line instead of
                # reading as the corruption it is.
                duration_s=float(meta["duration_s"]),
                # Zero, not the stored count: `attempts` is what this run spent,
                # and this run spent nothing. What the original render paid stays
                # in the sidecar, under `synthesized_attempts`.
                attempts=0,
                ok=True,
                coverage=float(meta["coverage"]),
                dropped_sentence=str(meta.get("dropped_sentence", "")),
                transcript=str(meta.get("transcript", "")),
                # Kept as stored: it describes how THIS AUDIO was made, which is
                # still true. `reused` says what the run did.
                recovered_by=str(meta.get("recovered_by", "")),
                word_diagnostics=tuple(str(c) for c in meta.get("word_diagnostics", ())),
                reused=True,
            )
        except Exception:
            return None
        else:
            self.usable += 1
            return result

    def put(self, key: str, result: ChunkResult, sample_rate: int) -> None:
        """Commit a verified take. The sidecar is written LAST and is the marker.

        Two files are not one atomic entry, so the order carries the guarantee: the
        audio lands first under a temporary name and is renamed into place, and only
        then does the sidecar appear. A process killed anywhere in between leaves a
        stray wav that no lookup can reach, rather than an entry whose verdict
        describes audio that was never finished.

        Two renders filing the SAME key at the same time can still interleave
        their commits and leave a sidecar describing the other one's audio. That
        is a digest mismatch, which is a miss, and the next successful write for
        that key repairs it — a bounded loss of one take. Making it airtight
        means naming files by content and keeping a superseded copy of every
        take on disk, which is a worse trade than one re-synthesis in a race
        nobody has hit.

        Overwriting an existing entry — a reroll — has the same bounded shape: a
        kill between the two replacements leaves the new audio under the old
        verdict, which is a mismatch, which is a miss. No ordering avoids it,
        because a reroll means to replace the entry; only keeping the superseded
        take would, at the same cost.
        """
        audio = np.ascontiguousarray(result.audio, dtype=np.float32)
        meta = {
            "format": FORMAT_VERSION,
            "key": key,
            "sample_rate": sample_rate,
            "digest": hashlib.blake2b(audio.tobytes(), digest_size=16).hexdigest(),
            "text": result.text,
            "duration_s": result.duration_s,
            "coverage": result.coverage,
            "dropped_sentence": result.dropped_sentence,
            "transcript": result.transcript,
            "recovered_by": result.recovered_by,
            "word_diagnostics": list(result.word_diagnostics),
            "synthesized_attempts": result.attempts,
        }
        try:
            import soundfile as sf

            self.root.mkdir(parents=True, exist_ok=True)
            wav = self.root / f"{key}.wav"
            # Per-process temp names: two renders filing the same key at once
            # would otherwise interleave through one pair of temp files and
            # commit a sidecar describing the other run's audio. That fails
            # closed (a digest mismatch is a miss) but costs a real take.
            tmp_wav = self.root / f"{key}.{os.getpid()}.wav.tmp"
            # `format` stated, because the temp name ends in .tmp and soundfile
            # otherwise infers the container from the extension and refuses.
            sf.write(str(tmp_wav), audio, sample_rate, subtype="FLOAT", format="WAV")
            os.replace(tmp_wav, wav)

            sidecar = self.root / f"{key}.json"
            tmp_meta = self.root / f"{key}.{os.getpid()}.json.tmp"
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_meta, sidecar)
            self.usable += 1
        except Exception:
            # Not just OSError: soundfile raises LibsndfileError (a RuntimeError)
            # of its own. The rule is about consequence, not about which library
            # failed — a store that cannot file a take must never take down a
            # render that synthesised correctly.
            self.write_failures += 1

