"""End-to-end pipeline: audio -> diarization -> transcription -> per-speaker summary."""

import json
from dataclasses import asdict
from pathlib import Path

from .audio import normalize_to_wav
from .diarize import diarize
from .summarize import Turn, format_transcript, summarize_per_speaker
from .transcribe import DEFAULT_MODEL, transcribe_segments


def run(
    audio: Path,
    out_dir: Path,
    num_speakers: int | None = None,
    model_card: str = DEFAULT_MODEL,
    skip_summary: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    print(f"[1/4] Normalizing audio -> 16kHz mono wav")
    wav = normalize_to_wav(audio, work / "session.wav")

    print(f"[2/4] Diarizing speakers (pyannote)" + (f", num_speakers={num_speakers}" if num_speakers else ""))
    segments = diarize(wav, num_speakers=num_speakers)
    print(f"      -> {len(segments)} segments, {len({s.speaker for s in segments})} speakers")

    print(f"[3/4] Transcribing segments in Tigrinya (model={model_card})")
    texts = transcribe_segments(wav, segments, tmp_dir=work / "segments", model_card=model_card)
    turns = [Turn(segment=s, text=t) for s, t in zip(segments, texts)]

    transcript_txt = format_transcript(turns)
    (out_dir / "transcript.txt").write_text(transcript_txt + "\n", encoding="utf-8")
    (out_dir / "transcript.json").write_text(
        json.dumps(
            [{"segment": asdict(t.segment), "text": t.text} for t in turns],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"      -> {out_dir / 'transcript.txt'}")

    if skip_summary:
        return {"transcript": str(out_dir / "transcript.txt"), "summaries": None}

    print(f"[4/4] Summarizing per speaker (Claude)")
    summaries = summarize_per_speaker(turns)
    summary_md = ["# Bible study session — per-speaker summary\n"]
    for speaker, text in summaries.items():
        summary_md.append(f"## {speaker}\n\n{text}\n")
    (out_dir / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    print(f"      -> {out_dir / 'summary.md'}")

    return {"transcript": str(out_dir / "transcript.txt"), "summaries": str(out_dir / "summary.md")}
