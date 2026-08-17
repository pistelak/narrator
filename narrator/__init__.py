"""Long-form narration from text.

Turns an ordered sequence of Text and Gap segments into one mastered audio file,
with per-chunk verification so that "it produced audio" and "it produced the
right audio" are different answers.

    from narrator import Text, Gap, Voice, render

    report = render(
        [Text("On the northern coast stands a lighthouse no ship will ever pass."),
         Gap(3.0),
         Text("Not the keeper. Not a stranger.")],
        voice=Voice(Path("voice.wav"), "reference transcript"),
        backend=HiggsBackend(),
        out=Path("episode.wav"),
    )  # verification defaults to the library's own policy, at the backend's rate
    if not report.clean:
        ...  # report.failures tells you which chunks and why
"""

from narrator.audio import MasterConfig
from narrator.prosody import yes_no_question
from narrator.render import RenderConfig, RenderFailed, render
from narrator.synth import SynthConfig
from narrator.types import (
    ASR,
    Audio,
    Backend,
    ChunkResult,
    Gap,
    RenderReport,
    Segment,
    Text,
    Verdict,
    Verifier,
    Voice,
)
from narrator.verify import CascadeVerifier, CoverageVerifier, NullVerifier, default_verifier

__version__ = "0.1.0"

__all__ = [
    "ASR",
    "Audio",
    "Backend",
    "CascadeVerifier",
    "ChunkResult",
    "CoverageVerifier",
    "Gap",
    "MasterConfig",
    "NullVerifier",
    "RenderConfig",
    "RenderFailed",
    "RenderReport",
    "Segment",
    "SynthConfig",
    "Text",
    "Verdict",
    "Verifier",
    "Voice",
    "__version__",
    "default_verifier",
    "render",
    "yes_no_question",
]
