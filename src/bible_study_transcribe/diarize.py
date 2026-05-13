"""Speaker diarization with pyannote.audio."""

import os
from pathlib import Path

from .audio import Segment


DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"


def diarize(wav_path: Path, num_speakers: int | None = None, hf_token: str | None = None) -> list[Segment]:
    """Return diarized speaker segments. Requires HF_TOKEN with model access accepted."""
    # Lazy import — pyannote pulls in heavy deps
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN required for pyannote. Get one at https://hf.co/settings/tokens "
            "and accept the model license at https://hf.co/pyannote/speaker-diarization-3.1"
        )

    pipeline = Pipeline.from_pretrained(DEFAULT_PIPELINE, use_auth_token=token)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **kwargs)

    segments: list[Segment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(Segment(start=turn.start, end=turn.end, speaker=speaker))
    return segments
