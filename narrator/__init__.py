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
        out=Path("episode.wav"),
    )
    if not report.clean:
        ...  # report.failures tells you which chunks and why
"""

from narrator.types import (
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

__version__ = "0.1.0"

__all__ = [
    "Audio", "Backend", "ChunkResult", "Gap", "RenderReport",
    "Segment", "Text", "Verdict", "Verifier", "Voice", "__version__",
]
