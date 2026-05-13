"""CLI entry point: bible-study-transcribe path/to/session.m4a"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import run
from .transcribe import DEFAULT_MODEL


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="bible-study-transcribe", description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to session audio (m4a/mp3/wav/...)")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--speakers", type=int, default=None, help="Hint number of speakers")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Omnilingual ASR model card")
    parser.add_argument("--no-summary", action="store_true", help="Skip the Claude summary step")
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"audio file not found: {args.audio}")

    session_out = args.out / args.audio.stem
    result = run(
        audio=args.audio,
        out_dir=session_out,
        num_speakers=args.speakers,
        model_card=args.model,
        skip_summary=args.no_summary,
    )
    print("\nDone.")
    print(f"  transcript: {result['transcript']}")
    if result["summaries"]:
        print(f"  summary:    {result['summaries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
