"""Audio normalization and segment slicing via ffmpeg."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_to_wav(src: Path, dst: Path) -> Path:
    """Convert any audio to 16 kHz mono PCM wav."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


def slice_segment(src_wav: Path, segment: Segment, dst: Path) -> Path:
    """Cut a single segment out of a normalized wav."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{segment.start:.3f}",
            "-to", f"{segment.end:.3f}",
            "-i", str(src_wav),
            "-c", "copy",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst
