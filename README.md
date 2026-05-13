# bible-study-transcribe

Tigrinya Bible-study sessions → diarized transcript → per-speaker English summary.

**Pipeline**
1. Normalize audio to 16 kHz mono wav (`ffmpeg`)
2. Speaker diarization with `pyannote/speaker-diarization-3.1`
3. Per-segment transcription with Meta's Omnilingual ASR (`tir_Ethi`)
4. Per-speaker English summary with Claude (Anthropic API, prompt-cached)

## Setup

```bash
cd ~/dev/bible-study-transcribe
uv sync
cp .env.example .env  # fill HF_TOKEN and ANTHROPIC_API_KEY
```

You must also:
- Accept the pyannote model licenses on Hugging Face (see `.env.example`).
- Have `ffmpeg` installed (`brew install ffmpeg`).

## Run

```bash
uv run bible-study-transcribe path/to/session.m4a --speakers 4
```

Outputs land in `outputs/<session-name>/`:
- `transcript.txt` — `[hh:mm:ss–hh:mm:ss] SPEAKER_xx: ...` in Tigrinya
- `transcript.json` — same data, structured
- `summary.md` — English summary per speaker

After it runs, relabel speakers in `transcript.txt` (e.g. `SPEAKER_00` → `Leader`) and re-run if you want updated summaries with the real names.

## Models

- ASR default: `omniASR_LLM_7B_v2` (~29 GB cached in `~/.cache/fairseq2/`)
  Drop to `omniASR_LLM_3B_v2` or `omniASR_LLM_1B_v2` for faster runs.
- Diarization: `pyannote/speaker-diarization-3.1`
- Summary: `claude-opus-4-7`

## Cost / time on M4 Pro (rough)

| Audio length | Diarization | ASR (7B) | Summary | Total |
|---|---|---|---|---|
| 30 sec | <10s | ~1 min | ~5s | ~1 min |
| 1 hour | ~5 min | ~30 min | ~30s | ~35 min |
| 2 hours | ~10 min | ~60 min | ~1 min | ~70 min |
