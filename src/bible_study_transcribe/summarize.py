"""Per-speaker English summary via Claude with prompt caching."""

import os
from collections import defaultdict
from dataclasses import dataclass

from .audio import Segment


MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are an assistant that summarizes Tigrinya Bible-study sessions.
You will receive a diarized transcript: a list of turns labeled by speaker (e.g. SPEAKER_00).
The transcript text is in Tigrinya (Ge'ez script). Tigrinya is a Semitic language spoken in Eritrea and northern Ethiopia.

For each speaker, produce a clear English summary covering:
- Main points and arguments they raised
- Scripture references they quoted (book/chapter/verse if mentioned)
- Questions they asked
- Any practical applications or commitments they made

Be faithful to the source — do not invent details. If a passage is unclear or ambiguous, say so."""


@dataclass
class Turn:
    segment: Segment
    text: str


def group_by_speaker(turns: list[Turn]) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        if t.text.strip():
            grouped[t.segment.speaker].append(t)
    return grouped


def format_transcript(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        ts = f"[{_fmt(t.segment.start)}–{_fmt(t.segment.end)}]"
        lines.append(f"{ts} {t.segment.speaker}: {t.text}")
    return "\n".join(lines)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def summarize_per_speaker(turns: list[Turn], api_key: str | None = None) -> dict[str, str]:
    """Return {speaker_id: english_summary}. Uses prompt caching on the full transcript."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    full_transcript = format_transcript(turns)
    speakers = sorted({t.segment.speaker for t in turns if t.text.strip()})

    summaries: dict[str, str] = {}
    for speaker in speakers:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                {
                    "type": "text",
                    "text": f"Full diarized transcript:\n\n{full_transcript}",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize what {speaker} contributed to the session. Output in English.",
                }
            ],
        )
        summaries[speaker] = resp.content[0].text.strip()
    return summaries
