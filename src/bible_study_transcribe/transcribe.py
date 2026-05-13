"""Tigrinya transcription via Meta's Omnilingual ASR."""

from pathlib import Path

from .audio import Segment, slice_segment


DEFAULT_MODEL = "omniASR_LLM_7B_v2"
TIGRINYA = "tir_Ethi"
MIN_SEGMENT_SECONDS = 0.4


def transcribe_segments(
    wav_path: Path,
    segments: list[Segment],
    tmp_dir: Path,
    model_card: str = DEFAULT_MODEL,
    batch_size: int = 4,
) -> list[str]:
    """Slice each segment, transcribe in Tigrinya, return parallel list of texts."""
    from omnilingual_asr.models.inference import ASRInferencePipeline

    pipeline = ASRInferencePipeline(model_card=model_card)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, seg in enumerate(segments):
        if seg.duration < MIN_SEGMENT_SECONDS:
            paths.append("")
            continue
        out = tmp_dir / f"seg_{i:05d}.wav"
        slice_segment(wav_path, seg, out)
        paths.append(str(out))

    real_paths = [p for p in paths if p]
    if not real_paths:
        return [""] * len(segments)

    texts = pipeline.transcribe(real_paths, lang=[TIGRINYA] * len(real_paths), batch_size=batch_size)

    out_texts: list[str] = []
    it = iter(texts)
    for p in paths:
        out_texts.append(next(it) if p else "")
    return out_texts
