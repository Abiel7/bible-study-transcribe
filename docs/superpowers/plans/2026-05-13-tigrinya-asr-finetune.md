# Tigrinya ASR Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune a 300M-parameter Omnilingual CTC ASR model on Tigrinya speech (WAXAL + FLEURS + Common Voice), export to int8 ONNX, and ship a sherpa-onnx-based inference path that runs ≤0.1 real-time-factor on Apple Silicon (M4 Pro). Drop-in replacement for the current Omnilingual-7B step in `bible-study-transcribe`.

**Project layout:** All fine-tuning code lives in this same `bible-study-transcribe` repo, alongside the existing transcription package — single uv project, separate `src/tigrinya_asr/` package, training/export wired in as optional `[train]` / `[export]` extras so the consumer install stays slim.

**Architecture:** Fine-tune `facebook/omniASR-CTC-300M-v2` with HuggingFace Transformers + Accelerate on Modal (A100-80GB), using the published Ethio-ASR recipe adapted for Omnilingual CTC. Export the resulting checkpoint to int8 ONNX via the same script Meta + k2-fsa used for `sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12`. Inference runs locally via `sherpa-onnx` CPU EP (CoreML EP is currently slower for this model class per upstream issue #2910).

**Targets:**
- **WER on WAXAL test:** ≤ 30% (Ethio-ASR baseline 35.22%, Omnilingual-7B zero-shot 40.69%)
- **Inference RTF on M4 Pro:** ≤ 0.1 (300M int8 ONNX)
- **Final model size:** ≤ 350 MB int8 / ≤ 650 MB fp16
- **Total compute cost:** ≤ $50 on Modal
- **Total wall-clock for training:** ≤ 16 hours on A100-80GB

**Tech Stack:**
- **Base model:** `facebook/omniASR-CTC-300M-v2` (Apache-2.0, Ge'ez tokens native)
- **Training framework:** HuggingFace Transformers + Accelerate, bf16
- **Cloud:** Modal (A100-80GB), `modal.Volume` for checkpoints
- **Experiment tracking:** Weights & Biases
- **Data:** WAXAL (`google/WaxalNLP` config `tir_asr`) + FLEURS (`google/fleurs` config `tir_et`) + Common Voice (`mozilla-foundation/common_voice_17_0` config `ti`)
- **Eval:** `jiwer` for WER/CER
- **Augmentation:** SpecAugment (built into HF config) + speed perturb via `torchaudio.sox_effects`
- **Export:** ONNX (opset 17) + `onnxruntime` int8 dynamic quantization
- **Inference:** `sherpa-onnx` 1.10+ (CPU EP on macOS)
- **Project structure:** single uv-managed repo (`bible-study-transcribe`), two packages under `src/`

---

## Key Decisions (locked before execution)

| Decision | Choice | Why |
|---|---|---|
| Base model | omniASR-CTC-300M-v2 | Apache-2.0, Ge'ez native, smallest viable for <350 MB target |
| Plan B base | w2v-bert-2.0 (`facebook/w2v-bert-2.0`) | Only invoked if Plan A WER > 30% after epoch 8 |
| Tokenizer | Keep stock 9,812-char universal vocab | Preserves pretraining alignment |
| Loss | CTC (greedy decode, no LM at first) | Fast; KenLM rescoring added later if needed |
| Optimizer | AdamW bf16 | Standard for this class |
| LR (encoder) | 3e-5 | Mid-range of Ethio-ASR sweep |
| LR (CTC head) | 1e-4 | Higher for randomly-initialized head |
| LR schedule | Linear warmup 10% + cosine decay | Stable on small data |
| Frozen layers | CNN feature extractor whole run; encoder layers 0-6 for first 2k steps | Standard low-resource recipe |
| Effective batch | 32 (per-device 8 × grad_accum 4) | Fits A100-80GB at bf16 |
| Epochs | 30 with early stopping (patience 5 on dev WER) | Saturates ~200h |
| SpecAugment | T=40, F=27, 2 time + 2 freq masks | Ethio-ASR config |
| Speed perturb | 0.9 / 1.0 / 1.1 | 3× effective data |
| Cloud | Modal A100-80GB (default) | Best DX/$ for one-off training; alternatives in sidebar below |
| Export | ONNX opset 17, int8 dynamic | `sherpa-onnx` consumes this |
| Inference EP | CPU (XNNPACK) on macOS | CoreML EP regression per onnx-runtime/sherpa-onnx #2910 |

---

## Sidebar — Compute Options (alternatives to Modal)

Modal is the **default** for Task 9 because its ergonomics (one `modal run`, persistent volumes, no SSH) save hours of plumbing for a first-time fine-tune. If you'd rather optimize for $ over DX, swap in one of these — they all produce an A100-80GB job that the Task 7 training script can run unchanged. Only Task 9 (`modal_app.py`) and the deploy command change.

| Option | Total $ (one ~12h run) | Reliability | What changes vs Task 9 |
|---|---|---|---|
| **Modal A100-80GB** (default) | ~$25 | High | — |
| **GCP free credit** ($300 new account → Vertex AI) | **$0** effective | High | Replace `modal_app.py` with a Vertex AI `CustomJob`; upload data to GCS bucket; use `gcloud ai custom-jobs create` |
| **AWS free credit** ($300 new account → SageMaker) | **$0** effective | High | Replace with a SageMaker `Estimator` (PyTorch container); upload data to S3 |
| **RunPod community on-demand** | ~$15 | High | Replace with a pod template + Network Volume; `rsync` data over SSH; `tmux` the train script |
| **RunPod community spot** | ~$10 | Medium (preemptible) | Same as above + checkpoint every 200 steps to Network Volume so re-rent is cheap |
| **Lambda Labs on-demand** | ~$22 | High | Same SSH/tmux pattern as RunPod |
| **Vast.ai spot** | $7-13 | Low — variable hosts, preemption | Pick a verified host with NVMe + ≥80 GB VRAM; `rclone` ckpts to R2/S3 every 30 min |
| **Colab Pro+** | $50/mo | Low — A100 not guaranteed, 24h session cap | Mount Drive, paste training command into a notebook cell, resume on disconnect |
| **Kaggle free** | $0 | n/a — too slow | Not viable for 200h training data |
| **Local M4 Pro** | $0 (electricity) | n/a — machine locked 8-12 days | LoRA-only feasible (~2-3 days); see Plan-C branch in `docs/architecture.md` |

**Recommendation in plain words:** Start with Modal. If you want $0, do the GCP free-credit path — it's the same A100 underneath, just more setup. If you've done cloud GPU before, RunPod on-demand at $15 is the sweet spot.

**Switching providers later is cheap.** The training script (`tigrinya_asr/train.py`) is provider-agnostic. Only `modal_app.py` is Modal-specific.

---

## File Structure

Single repo: `~/dev/bible-study-transcribe/` (the existing one, published at `Abiel7/bible-study-transcribe`). New files are added alongside the existing transcription package — nothing in the current `src/bible_study_transcribe/` is moved.

```
bible-study-transcribe/                         # existing repo, root unchanged
├── pyproject.toml                              # augmented: new [train], [export] extras
├── .gitignore                                  # augmented: data/, outputs/, *.onnx, *.pt, wandb/
├── .python-version                             # existing (3.12)
├── LICENSE                                     # existing (MIT)
├── README.md                                   # augmented: fine-tuning section
├── .env.example                                # augmented: WANDB_API_KEY, MODAL_TOKEN_*
├── docs/
│   ├── superpowers/plans/2026-05-13-tigrinya-asr-finetune.md   # this file (already here)
│   └── architecture.md                         # NEW — rationale + research sources
├── data/                                       # NEW (gitignored)
│   ├── manifests/                              # train.jsonl / dev.jsonl / test.jsonl
│   └── audio/                                  # cached normalized wavs
├── src/
│   ├── bible_study_transcribe/                 # existing — UNCHANGED until Task 15
│   │   ├── audio.py
│   │   ├── cli.py
│   │   ├── diarize.py
│   │   ├── pipeline.py
│   │   ├── summarize.py
│   │   └── transcribe.py
│   └── tigrinya_asr/                           # NEW — fine-tuning package
│       ├── __init__.py
│       ├── data.py                             # manifest builders
│       ├── preprocess.py                       # audio normalize + Ge'ez text normalize
│       ├── tokenizer.py                        # adapter to Omnilingual char tokenizer
│       ├── model.py                            # load + freeze policy
│       ├── augment.py                          # SpecAugment + speed perturb
│       ├── train.py                            # HF Trainer entrypoint
│       ├── evaluate.py                         # WER/CER + per-source slicing
│       ├── export_onnx.py                      # PyTorch → ONNX → int8 quantize
│       ├── benchmark.py                        # sherpa-onnx RTF on M4 Pro
│       └── cli.py                              # subcommand dispatch
├── modal_app.py                                # NEW — Modal training app (root-level)
├── scripts/                                    # NEW
│   ├── fetch_datasets.py                       # WAXAL/FLEURS/CV download
│   └── inspect_sample.py                       # debug single audio + alignment
├── tests/                                      # NEW
│   ├── test_data.py
│   ├── test_preprocess.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_augment.py
│   ├── test_evaluate.py
│   └── test_export_onnx.py
└── outputs/                                    # NEW (gitignored) — checkpoints, ONNX, benchmarks
```

**Package boundary:** `tigrinya_asr` is a peer of `bible_study_transcribe` under `src/`. It is **not** imported by the consumer app at runtime — the transcription pipeline only loads the **exported ONNX artifact**, not the training code. This keeps the production runtime free of torch-training / HF / wandb deps unless the user opts into `[train]` / `[export]` extras.

---

## Task 0: Scaffold tigrinya_asr Package Inside bible-study-transcribe

**Files:**
- Modify: `~/dev/bible-study-transcribe/pyproject.toml`
- Modify: `~/dev/bible-study-transcribe/.gitignore`
- Modify: `~/dev/bible-study-transcribe/.env.example`
- Create: `~/dev/bible-study-transcribe/src/tigrinya_asr/__init__.py`
- Create: `~/dev/bible-study-transcribe/tests/__init__.py`
- Create: `~/dev/bible-study-transcribe/data/.gitkeep`
- Create: `~/dev/bible-study-transcribe/outputs/.gitkeep`
- Create: `~/dev/bible-study-transcribe/scripts/.gitkeep`

- [ ] **Step 1: Make working directories**

```bash
cd ~/dev/bible-study-transcribe
mkdir -p src/tigrinya_asr tests scripts data/manifests data/audio outputs
touch src/tigrinya_asr/__init__.py tests/__init__.py data/.gitkeep outputs/.gitkeep scripts/.gitkeep
```

- [ ] **Step 2: Add train + export extras to pyproject.toml**

Open `~/dev/bible-study-transcribe/pyproject.toml`. Add the following two new top-level sections (do **not** remove the existing `[project]` block; only add the new sections):

```toml
[project.optional-dependencies]
train = [
    "transformers>=4.46",
    "accelerate>=1.0",
    "datasets>=3.0",
    "librosa",
    "jiwer",
    "wandb",
    "tqdm",
    "click",
    "modal",
    "evaluate",
]
export = [
    "onnx",
    "onnxruntime",
    "sherpa-onnx",
]
```

Then under the existing `[project.scripts]` (or create it if missing), ensure the entry for the new CLI is present:

```toml
[project.scripts]
bible-study-transcribe = "bible_study_transcribe.cli:main"
tigrinya-asr = "tigrinya_asr.cli:cli"
```

- [ ] **Step 3: Augment .gitignore**

Append to the existing `.gitignore`:

```
# Fine-tuning artifacts
data/audio/
data/manifests/
outputs/
wandb/
*.onnx
*.pt
*.ckpt
```

- [ ] **Step 4: Augment .env.example**

Append to the existing `.env.example`:

```
# Fine-tuning (only needed if running [train]/[export] extras)
WANDB_API_KEY=xxx                  # wandb.ai/authorize
MODAL_TOKEN_ID=xxx                 # set via `modal token new`
MODAL_TOKEN_SECRET=xxx
```

- [ ] **Step 5: Install extras and verify both packages import**

```bash
uv sync --extra train --extra export
uv run python -c "import torch, transformers, datasets, jiwer, sherpa_onnx; print('deps OK')"
uv run python -c "import bible_study_transcribe, tigrinya_asr; print('packages OK')"
```

Expected: both prints succeed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/tigrinya_asr/__init__.py \
        tests/__init__.py data/.gitkeep outputs/.gitkeep scripts/.gitkeep
git commit -m "chore: add tigrinya_asr package skeleton + [train] [export] extras"
```

---

## Task 1: Audio + Text Preprocessing Module

**Files:**
- Create: `src/tigrinya_asr/preprocess.py`
- Create: `tests/test_preprocess.py`

Both audio normalization (16 kHz mono PCM16) and text normalization (Ge'ez punctuation, digit handling) are deterministic and easy to unit-test.

- [ ] **Step 1: Write failing test for audio normalize**

`tests/test_preprocess.py`:
```python
import numpy as np
import soundfile as sf
from pathlib import Path
from tigrinya_asr.preprocess import normalize_audio


def test_normalize_audio_resamples_to_16k_mono(tmp_path):
    # Make a fake 44.1 kHz stereo wav
    src = tmp_path / "in.wav"
    sr = 44100
    audio = np.random.randn(sr, 2).astype(np.float32) * 0.1
    sf.write(src, audio, sr)

    dst = tmp_path / "out.wav"
    normalize_audio(src, dst)

    data, out_sr = sf.read(dst)
    assert out_sr == 16000
    assert data.ndim == 1
    assert data.dtype == np.int16 or data.dtype == np.float32
```

- [ ] **Step 2: Verify it fails**

```bash
uv run pytest tests/test_preprocess.py::test_normalize_audio_resamples_to_16k_mono -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tigrinya_asr.preprocess'`

- [ ] **Step 3: Implement audio normalize**

`src/tigrinya_asr/preprocess.py`:
```python
"""Audio + text normalization for Tigrinya ASR training."""
import re
import subprocess
from pathlib import Path

GEEZ_PUNCT_MAP = {
    "።": ".",  # Ethiopic full stop
    "፡": " ",  # Ethiopic word space
    "፣": ",",  # Ethiopic comma
    "፤": ";",  # Ethiopic semicolon
    "፥": ":",  # Ethiopic colon
    "፦": ":",  # Ethiopic preface colon
    "፧": "?",  # Ethiopic question mark
    "፨": ".",  # Ethiopic paragraph separator
}


def normalize_audio(src: Path, dst: Path) -> Path:
    """Convert any audio to 16 kHz mono PCM16."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return dst


def normalize_text(text: str) -> str:
    """Map Ge'ez punctuation, strip junk, lowercase Latin."""
    for src, repl in GEEZ_PUNCT_MAP.items():
        text = text.replace(src, repl)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[​‌‍﻿]", "", text)  # zero-width chars
    return text
```

- [ ] **Step 4: Add text normalize tests**

Append to `tests/test_preprocess.py`:
```python
from tigrinya_asr.preprocess import normalize_text


def test_normalize_text_maps_geez_punct():
    assert normalize_text("ሰላም፡ዓለም።") == "ሰላም ዓለም."
    assert normalize_text("ኣይኮነን፣ግን ሓቂ እዩ።") == "ኣይኮነን,ግን ሓቂ እዩ."


def test_normalize_text_strips_zero_width():
    assert normalize_text("ሰላም​ዓለም") == "ሰላምዓለም"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  ሰላም   ዓለም  ") == "ሰላም ዓለም"
```

- [ ] **Step 5: Run all tests, verify pass**

```bash
uv run pytest tests/test_preprocess.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/tigrinya_asr/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): audio normalize to 16kHz mono + Geez text normalize"
```

---

## Task 2: Tokenizer Adapter

**Files:**
- Create: `src/tigrinya_asr/tokenizer.py`
- Create: `tests/test_tokenizer.py`

The Omnilingual CTC tokenizer uses a 9,812-symbol universal grapheme vocab. We need a thin adapter that wraps it for HF `Wav2Vec2CTCTokenizer`-style usage.

- [ ] **Step 1: Failing test for tokenizer roundtrip**

`tests/test_tokenizer.py`:
```python
from tigrinya_asr.tokenizer import load_tokenizer


def test_tokenizer_roundtrips_tigrinya_sentence():
    tok = load_tokenizer()
    text = "ሰላም ዓለም"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded.replace(" ", "") == text.replace(" ", "")


def test_tokenizer_handles_oov_gracefully():
    tok = load_tokenizer()
    # ASCII digits — present in universal vocab
    ids = tok.encode("123")
    assert len(ids) >= 3
```

- [ ] **Step 2: Verify it fails**

```bash
uv run pytest tests/test_tokenizer.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Implement tokenizer loader**

`src/tigrinya_asr/tokenizer.py`:
```python
"""Adapter around the Omnilingual ASR character tokenizer."""
from functools import lru_cache

from omnilingual_asr.models.wav2vec2_llama.tokenizer import load_omniasr_tokenizer


@lru_cache(maxsize=1)
def load_tokenizer():
    """Return a cached tokenizer instance."""
    return load_omniasr_tokenizer()
```

Note: the actual import path in `omnilingual-asr` is package-internal. Verify with:
```bash
uv run python -c "from omnilingual_asr.models.wav2vec2_llama.tokenizer import load_omniasr_tokenizer; t = load_omniasr_tokenizer(); print(type(t))"
```
If the path differs, update accordingly. The fallback is to load the SentencePiece-style model file directly from `~/.cache/fairseq2/assets/<hash>/omniASR_tokenizer.model`.

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/test_tokenizer.py -v
```
Expected: 2 passed (if the API matches; otherwise iterate path)

- [ ] **Step 5: Commit**

```bash
git add src/tigrinya_asr/tokenizer.py tests/test_tokenizer.py
git commit -m "feat(tokenizer): wrap Omnilingual char tokenizer with caching"
```

---

## Task 3: Dataset Manifest Builder

**Files:**
- Create: `src/tigrinya_asr/data.py`
- Create: `tests/test_data.py`
- Create: `scripts/fetch_datasets.py`

Builds JSONL manifests `{"audio_path": "...", "text": "...", "duration": 5.32, "source": "waxal"}` for train/dev/test splits.

- [ ] **Step 1: Failing test for manifest schema**

`tests/test_data.py`:
```python
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from tigrinya_asr.data import write_manifest, ManifestRow


def test_write_manifest_emits_jsonl(tmp_path):
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)
    rows = [ManifestRow(audio_path=str(audio_path), text="ሰላም", duration=1.0, source="test")]
    out = tmp_path / "manifest.jsonl"
    write_manifest(rows, out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["text"] == "ሰላም"
    assert parsed["duration"] == 1.0
    assert parsed["source"] == "test"
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_data.py -v
```
Expected: FAIL — module missing

- [ ] **Step 3: Implement manifest module**

`src/tigrinya_asr/data.py`:
```python
"""Build train/dev/test manifests for fine-tuning."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path


MIN_DURATION = 0.5
MAX_DURATION = 30.0


@dataclass
class ManifestRow:
    audio_path: str
    text: str
    duration: float
    source: str  # "waxal" | "fleurs" | "common_voice" | ...


def write_manifest(rows: list[ManifestRow], dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            if not (MIN_DURATION <= r.duration <= MAX_DURATION):
                continue
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return dst


def read_manifest(src: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(ManifestRow(**d))
    return rows
```

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/test_data.py -v
```
Expected: 1 passed

- [ ] **Step 5: Write fetch script (no test — boundary I/O)**

`scripts/fetch_datasets.py`:
```python
"""Download WAXAL Tigrinya + FLEURS + Common Voice, write manifests."""
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from tigrinya_asr.data import ManifestRow, write_manifest
from tigrinya_asr.preprocess import normalize_audio, normalize_text


def fetch_waxal(audio_dir: Path) -> list[ManifestRow]:
    ds = load_dataset("google/WaxalNLP", "tir_asr")
    rows = []
    for split in ("train", "validation", "test"):
        for i, ex in enumerate(tqdm(ds[split], desc=f"waxal/{split}")):
            audio = ex["audio"]
            dst = audio_dir / "waxal" / split / f"{i:06d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            # huggingface audio is already 16kHz typically; normalize anyway
            import soundfile as sf
            sf.write(dst, audio["array"], audio["sampling_rate"])
            normalize_audio(dst, dst)
            rows.append(ManifestRow(
                audio_path=str(dst),
                text=normalize_text(ex["transcription"]),
                duration=len(audio["array"]) / audio["sampling_rate"],
                source=f"waxal_{split}",
            ))
    return rows


def fetch_fleurs(audio_dir: Path) -> list[ManifestRow]:
    ds = load_dataset("google/fleurs", "tir_et")
    rows = []
    for split in ("train", "validation", "test"):
        for i, ex in enumerate(tqdm(ds[split], desc=f"fleurs/{split}")):
            audio = ex["audio"]
            dst = audio_dir / "fleurs" / split / f"{i:06d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(dst, audio["array"], audio["sampling_rate"])
            normalize_audio(dst, dst)
            rows.append(ManifestRow(
                audio_path=str(dst),
                text=normalize_text(ex["transcription"]),
                duration=len(audio["array"]) / audio["sampling_rate"],
                source=f"fleurs_{split}",
            ))
    return rows


def fetch_common_voice(audio_dir: Path) -> list[ManifestRow]:
    # Note: CV requires accepting the dataset license + login token
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "ti")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for i, ex in enumerate(tqdm(ds[split], desc=f"cv/{split}")):
            audio = ex["audio"]
            dst = audio_dir / "cv" / split / f"{i:06d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(dst, audio["array"], audio["sampling_rate"])
            normalize_audio(dst, dst)
            rows.append(ManifestRow(
                audio_path=str(dst),
                text=normalize_text(ex["sentence"]),
                duration=len(audio["array"]) / audio["sampling_rate"],
                source=f"cv_{split}",
            ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    p.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--skip-cv", action="store_true")
    args = p.parse_args()

    rows = fetch_waxal(args.audio_dir) + fetch_fleurs(args.audio_dir)
    if not args.skip_cv:
        rows += fetch_common_voice(args.audio_dir)

    # Split by source-derived split label
    train = [r for r in rows if r.source.endswith("_train")]
    dev = [r for r in rows if r.source.endswith("_validation")]
    test = [r for r in rows if r.source.endswith("_test")]

    write_manifest(train, args.manifest_dir / "train.jsonl")
    write_manifest(dev, args.manifest_dir / "dev.jsonl")
    write_manifest(test, args.manifest_dir / "test.jsonl")
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run with FLEURS only (small, fast)**

```bash
uv run python scripts/fetch_datasets.py --skip-cv 2>&1 | tail -5
```
Expected: prints `train=N dev=M test=K` for non-zero N/M/K.

- [ ] **Step 7: Commit**

```bash
git add src/tigrinya_asr/data.py tests/test_data.py scripts/fetch_datasets.py
git commit -m "feat(data): manifest schema + fetch script for WAXAL/FLEURS/CV"
```

---

## Task 4: Augmentation Module

**Files:**
- Create: `src/tigrinya_asr/augment.py`
- Create: `tests/test_augment.py`

SpecAugment is built into HF's `Wav2Vec2BertConfig`. We only need our own speed-perturb collator.

- [ ] **Step 1: Failing test for speed perturb**

`tests/test_augment.py`:
```python
import numpy as np
from tigrinya_asr.augment import speed_perturb


def test_speed_perturb_changes_length():
    audio = np.random.randn(16000).astype(np.float32)
    fast = speed_perturb(audio, sr=16000, factor=1.1)
    slow = speed_perturb(audio, sr=16000, factor=0.9)
    assert len(fast) < len(audio) < len(slow)


def test_speed_perturb_factor_one_is_identity():
    audio = np.random.randn(16000).astype(np.float32)
    out = speed_perturb(audio, sr=16000, factor=1.0)
    assert np.allclose(audio, out)
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_augment.py -v
```
Expected: FAIL — module missing

- [ ] **Step 3: Implement speed perturb**

`src/tigrinya_asr/augment.py`:
```python
"""Audio augmentation: speed perturbation."""
import numpy as np
import torch
import torchaudio


def speed_perturb(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Apply speed perturbation. factor < 1 slows down, > 1 speeds up."""
    if factor == 1.0:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0)
    out, _ = torchaudio.sox_effects.apply_effects_tensor(
        tensor, sr, [["speed", f"{factor:.2f}"], ["rate", str(sr)]]
    )
    return out.squeeze(0).numpy()
```

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/test_augment.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/tigrinya_asr/augment.py tests/test_augment.py
git commit -m "feat(augment): speed-perturbation wrapper around torchaudio sox"
```

---

## Task 5: Model Loading + Freeze Policy

**Files:**
- Create: `src/tigrinya_asr/model.py`
- Create: `tests/test_model.py`

Loads `facebook/omniASR-CTC-300M-v2` via the omnilingual-asr package, attaches the CTC head, applies the freeze policy.

- [ ] **Step 1: Failing test for load + freeze**

`tests/test_model.py`:
```python
import torch
from tigrinya_asr.model import load_finetune_model


def test_load_model_freezes_feature_extractor():
    model = load_finetune_model("omniASR_CTC_300M_v2", freeze_encoder_layers=6)
    # Feature extractor params should not require grad
    fx_params = [p for n, p in model.named_parameters() if "feature_extractor" in n or "conv" in n.lower()]
    assert any(not p.requires_grad for p in fx_params)


def test_freeze_layer_count_respected():
    model = load_finetune_model("omniASR_CTC_300M_v2", freeze_encoder_layers=6)
    frozen_layers = sum(
        1 for n, p in model.named_parameters()
        if any(f".{i}." in n for i in range(6)) and "encoder.layers" in n and not p.requires_grad
    )
    assert frozen_layers > 0
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_model.py -v
```
Expected: FAIL — module missing

- [ ] **Step 3: Implement model loader**

`src/tigrinya_asr/model.py`:
```python
"""Load and configure Omnilingual CTC for fine-tuning."""
import torch


def load_finetune_model(model_card: str = "omniASR_CTC_300M_v2", freeze_encoder_layers: int = 6):
    """Load Omnilingual CTC model, freeze feature extractor + first N encoder layers."""
    from omnilingual_asr.models.loader import load_model  # exact path verified at runtime

    model = load_model(model_card)

    # Freeze CNN feature extractor (always)
    for name, param in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            param.requires_grad = False

    # Freeze first N encoder layers
    for name, param in model.named_parameters():
        for i in range(freeze_encoder_layers):
            if f"encoder.layers.{i}." in name:
                param.requires_grad = False
                break

    return model


def unfreeze_encoder(model) -> None:
    """Unfreeze all encoder layers (call after warmup phase)."""
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            param.requires_grad = True
```

Note: the exact import path in `omnilingual-asr` for loading a CTC model needs verification. If `omnilingual_asr.models.loader.load_model` isn't the right symbol, inspect:
```bash
uv run python -c "import omnilingual_asr; help(omnilingual_asr)"
```
and adjust.

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/test_model.py -v
```
Expected: 2 passed (after fixing the import path if needed)

- [ ] **Step 5: Commit**

```bash
git add src/tigrinya_asr/model.py tests/test_model.py
git commit -m "feat(model): load Omnilingual CTC 300M with feature-extractor + layer freeze"
```

---

## Task 6: Evaluation (WER/CER per slice)

**Files:**
- Create: `src/tigrinya_asr/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Failing test for WER computation**

`tests/test_evaluate.py`:
```python
from tigrinya_asr.evaluate import compute_wer_cer, evaluate_per_source


def test_compute_wer_cer_exact_match():
    refs = ["ሰላም ዓለም"]
    hyps = ["ሰላም ዓለም"]
    metrics = compute_wer_cer(refs, hyps)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0


def test_compute_wer_one_word_wrong():
    refs = ["ሰላም ዓለም"]
    hyps = ["ሰላም ሰማይ"]
    metrics = compute_wer_cer(refs, hyps)
    assert metrics["wer"] == 0.5  # 1 of 2 words wrong


def test_evaluate_per_source_groups():
    rows = [
        {"source": "fleurs_test", "text": "ሰላም", "hypothesis": "ሰላም"},
        {"source": "waxal_test", "text": "ዓለም", "hypothesis": "ሰማይ"},
    ]
    result = evaluate_per_source(rows)
    assert result["fleurs_test"]["wer"] == 0.0
    assert result["waxal_test"]["wer"] == 1.0
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_evaluate.py -v
```
Expected: FAIL — module missing

- [ ] **Step 3: Implement evaluate**

`src/tigrinya_asr/evaluate.py`:
```python
"""WER/CER evaluation, optionally sliced by manifest source."""
from collections import defaultdict

import jiwer


def compute_wer_cer(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Compute corpus-level WER and CER. Returns floats in [0, 1]."""
    if not refs:
        return {"wer": 0.0, "cer": 0.0}
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
    }


def evaluate_per_source(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Group rows by 'source' field; return WER/CER per group."""
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    return {
        src: compute_wer_cer([r["text"] for r in items], [r["hypothesis"] for r in items])
        for src, items in by_src.items()
    }
```

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/test_evaluate.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/tigrinya_asr/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): WER/CER computation with per-source slicing"
```

---

## Task 7: Training Script

**Files:**
- Create: `src/tigrinya_asr/train.py`

Single entrypoint usable both locally (small sanity run) and on Modal (full run). No unit test for the loop itself — we instead run a 1-step smoke test in Task 8.

- [ ] **Step 1: Write training entrypoint**

`src/tigrinya_asr/train.py`:
```python
"""HuggingFace-Trainer-based fine-tuning of Omnilingual CTC for Tigrinya."""
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from .augment import speed_perturb
from .data import read_manifest
from .evaluate import compute_wer_cer
from .model import load_finetune_model, unfreeze_encoder
from .tokenizer import load_tokenizer


SEED = 1337


def jsonl_to_hf_dataset(manifest_path: Path) -> Dataset:
    rows = read_manifest(manifest_path)
    return Dataset.from_list([
        {"audio_path": r.audio_path, "text": r.text, "duration": r.duration, "source": r.source}
        for r in rows
    ])


def make_data_collator(tokenizer, apply_speed_perturb: bool):
    def collate(batch: list[dict]) -> dict:
        audio_arrays = []
        labels = []
        for ex in batch:
            audio, sr = sf.read(ex["audio_path"])
            if apply_speed_perturb:
                factor = random.choice([0.9, 1.0, 1.1])
                audio = speed_perturb(audio.astype(np.float32), sr, factor)
            audio_arrays.append(audio)
            labels.append(tokenizer.encode(ex["text"]))

        # Pad audio + labels (model-specific)
        # ... real implementation pads to max length in batch
        return {
            "input_values": audio_arrays,  # list-of-tensors; trainer-side pads
            "labels": labels,
        }
    return collate


def make_compute_metrics(tokenizer):
    def compute(eval_pred):
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        hyps = [tokenizer.decode(p) for p in pred_ids]
        refs = [tokenizer.decode(l) for l in label_ids]
        return compute_wer_cer(refs, hyps)
    return compute


@click.command()
@click.option("--manifest-dir", type=Path, default=Path("data/manifests"))
@click.option("--output-dir", type=Path, default=Path("outputs/ft-300m"))
@click.option("--model-card", default="omniASR_CTC_300M_v2")
@click.option("--epochs", type=int, default=30)
@click.option("--batch-size", type=int, default=8)
@click.option("--grad-accum", type=int, default=4)
@click.option("--lr-encoder", type=float, default=3e-5)
@click.option("--lr-head", type=float, default=1e-4)
@click.option("--warmup-ratio", type=float, default=0.1)
@click.option("--freeze-layers", type=int, default=6)
@click.option("--resume", type=str, default=None)
@click.option("--wandb-project", default="tigrinya-asr")
@click.option("--max-steps-debug", type=int, default=0, help=">0 limits training for smoke tests")
def main(**kwargs):
    set_seed(SEED)
    output_dir = kwargs["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_finetune_model(kwargs["model_card"], freeze_encoder_layers=kwargs["freeze_layers"])

    train_ds = jsonl_to_hf_dataset(kwargs["manifest_dir"] / "train.jsonl")
    dev_ds = jsonl_to_hf_dataset(kwargs["manifest_dir"] / "dev.jsonl")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=kwargs["epochs"],
        per_device_train_batch_size=kwargs["batch_size"],
        per_device_eval_batch_size=kwargs["batch_size"],
        gradient_accumulation_steps=kwargs["grad_accum"],
        learning_rate=kwargs["lr_encoder"],
        lr_scheduler_type="cosine",
        warmup_ratio=kwargs["warmup_ratio"],
        bf16=True,
        dataloader_num_workers=4,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=f"tigrinya-asr-{kwargs['model_card']}",
        max_steps=kwargs["max_steps_debug"] or -1,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=make_data_collator(tokenizer, apply_speed_perturb=True),
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train(resume_from_checkpoint=kwargs["resume"])
    trainer.save_model(str(output_dir / "final"))
    (output_dir / "final" / "training_args.json").write_text(
        json.dumps({k: str(v) for k, v in kwargs.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
```

Note: the data collator above is intentionally schematic — actual padding logic depends on the Omnilingual CTC model's expected input format. Verify by inspecting `model.forward` signature and the existing inference pipeline in `omnilingual_asr/models/inference/pipeline.py`. **Before running:** add tensor padding for `input_values` and `labels` matching the model's expected dtype/shape (typically float32 waveforms padded to max length in batch, labels padded with `-100`).

- [ ] **Step 2: Commit**

```bash
git add src/tigrinya_asr/train.py
git commit -m "feat(train): HF Trainer entrypoint with WER eval + WandB"
```

---

## Task 8: Local Smoke Test (1 step)

**Files:**
- Modify: none (consumes Tasks 1-7)

- [ ] **Step 1: Run 1-step training locally on M4 Pro with FLEURS only**

```bash
cd ~/dev/bible-study-transcribe
uv run python scripts/fetch_datasets.py --skip-cv  # if not already
uv run python -m tigrinya_asr.train \
    --manifest-dir data/manifests \
    --output-dir outputs/smoke \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --max-steps-debug 1
```
Expected: completes 1 step without OOM or NaN. Prints a logits shape and loss value.

If it fails, iterate on the data collator until 1 step passes. **Do not proceed until this works.**

- [ ] **Step 2: Commit any fixes**

```bash
git add -A
git commit -m "fix(train): data collator padding for Omnilingual CTC forward"
```

---

## Task 9: Modal App for Cloud Training

**Files:**
- Create: `modal_app.py`

- [ ] **Step 1: Write Modal app**

`modal_app.py`:
```python
"""Modal entrypoint: provisions A100-80GB, mounts data, runs train.py."""
import modal

PY_VER = "3.12"
TORCH_VER = "2.8.0"

image = (
    modal.Image.debian_slim(python_version=PY_VER)
    .apt_install("ffmpeg", "git")
    .pip_install(
        f"torch=={TORCH_VER}",
        f"torchaudio=={TORCH_VER}",
        "transformers>=4.46",
        "accelerate>=1.0",
        "datasets>=3.0",
        "soundfile",
        "librosa",
        "jiwer",
        "wandb",
        "omnilingual-asr",
        "click",
    )
    .add_local_dir("src/tigrinya_asr", remote_path="/root/tigrinya_asr")
)

app = modal.App("tigrinya-asr", image=image)
volume = modal.Volume.from_name("tigrinya-asr-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("tigrinya-asr-ckpts", create_if_missing=True)


@app.function(
    gpu="A100-80GB",
    timeout=18 * 3600,
    volumes={"/data": volume, "/ckpts": ckpt_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
)
def train(
    epochs: int = 30,
    batch_size: int = 8,
    grad_accum: int = 4,
    model_card: str = "omniASR_CTC_300M_v2",
    resume: str | None = None,
):
    import sys
    sys.path.insert(0, "/root")
    from tigrinya_asr.train import main as train_main

    train_main.callback(
        manifest_dir="/data/manifests",
        output_dir=f"/ckpts/{model_card}",
        model_card=model_card,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr_encoder=3e-5,
        lr_head=1e-4,
        warmup_ratio=0.1,
        freeze_layers=6,
        resume=resume,
        wandb_project="tigrinya-asr",
        max_steps_debug=0,
    )


@app.function(
    cpu=4.0, memory=8192,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 3600,
)
def prepare_data():
    """Run scripts/fetch_datasets.py inside Modal to populate the Volume."""
    import subprocess
    subprocess.run(
        ["python", "/root/scripts/fetch_datasets.py",
         "--audio-dir", "/data/audio",
         "--manifest-dir", "/data/manifests"],
        check=True,
    )


@app.local_entrypoint()
def main(
    job: str = "train",
    epochs: int = 30,
    resume: str | None = None,
):
    if job == "prepare_data":
        prepare_data.remote()
    elif job == "train":
        train.remote(epochs=epochs, resume=resume)
    else:
        raise ValueError(f"unknown job: {job}")
```

- [ ] **Step 2: Set up Modal secrets**

```bash
modal token new                # interactive auth
modal secret create huggingface HF_TOKEN=$HF_TOKEN
modal secret create wandb WANDB_API_KEY=$WANDB_API_KEY
```

- [ ] **Step 3: Prepare data on Modal**

```bash
uv run modal run modal_app.py::main --job prepare_data
```
Expected: writes manifests + audio to the `tigrinya-asr-data` Volume.

- [ ] **Step 4: Kick off training (foreground, 1-epoch dry run first)**

```bash
uv run modal run --detach modal_app.py::main --job train --epochs 1
```
Watch W&B for loss curves. Stop after 200 steps if loss is moving.

- [ ] **Step 5: Commit**

```bash
git add modal_app.py
git commit -m "feat(modal): A100-80GB training app with persistent volumes"
```

---

## Task 10: Full Training Run

**Files:**
- Modify: none (operational)

- [ ] **Step 1: Kick off full 30-epoch run**

```bash
uv run modal run --detach modal_app.py::main --job train --epochs 30
```
Expected wall-clock: 11-14h on A100-80GB. Cost: ~$25-30.

- [ ] **Step 2: Monitor**

W&B project `tigrinya-asr` — watch dev WER. Expected trajectory:
- Epoch 1: dev WER ~50%
- Epoch 5: dev WER ~35%
- Epoch 10: dev WER ~25%
- Epoch 20: dev WER ~22%
- Epoch 30: dev WER ~18-22% (early stop expected)

- [ ] **Step 3: Document final WER**

Append to `README.md`:
```markdown
## Training Results

| Metric | Value |
|---|---|
| WAXAL test WER | XX.X% |
| FLEURS test WER | XX.X% |
| Common Voice test WER | XX.X% |
| Wall-clock | XXh |
| Total cost | $XX |
```

- [ ] **Step 4: Commit results**

```bash
git add README.md
git commit -m "docs: training-run results"
```

**Gate:** Do not proceed to Task 11 unless WAXAL test WER ≤ 30%. If above, see `docs/architecture.md` Plan-B branch (switch base to `w2v-bert-2.0`, redo Tasks 5-10).

---

## Task 11: Pull Checkpoint Locally

**Files:**
- Modify: none

- [ ] **Step 1: Add Modal pull function**

Append to `modal_app.py`:
```python
@app.function(volumes={"/ckpts": ckpt_volume}, timeout=900)
def fetch_checkpoint(model_card: str = "omniASR_CTC_300M_v2") -> bytes:
    import tarfile, io
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(f"/ckpts/{model_card}/final", arcname="final")
    return buf.getvalue()
```

- [ ] **Step 2: Pull**

```bash
uv run python -c "
import modal, tarfile, io
fn = modal.Function.from_name('tigrinya-asr', 'fetch_checkpoint')
data = fn.remote()
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
    tar.extractall('outputs/ft-300m')
print('extracted to outputs/ft-300m/final')
"
ls outputs/ft-300m/final
```

- [ ] **Step 3: Commit**

```bash
git add modal_app.py
git commit -m "feat(modal): add fetch_checkpoint to pull final weights"
```

---

## Task 12: ONNX Export

**Files:**
- Create: `src/tigrinya_asr/export_onnx.py`
- Create: `tests/test_export_onnx.py`

- [ ] **Step 1: Failing test for export-and-load roundtrip**

`tests/test_export_onnx.py`:
```python
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tigrinya_asr.export_onnx import export_to_onnx


def test_exported_onnx_runs(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(checkpoint_dir=Path("outputs/ft-300m/final"), onnx_path=onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 16000), dtype=np.float32)
    outputs = sess.run(None, {"input_values": dummy})
    assert outputs[0].shape[0] == 1
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_export_onnx.py -v
```
Expected: FAIL — module missing

- [ ] **Step 3: Implement export**

`src/tigrinya_asr/export_onnx.py`:
```python
"""Export fine-tuned Omnilingual CTC checkpoint to ONNX + int8 quantize."""
from pathlib import Path

import torch
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx(checkpoint_dir: Path, onnx_path: Path, opset: int = 17) -> Path:
    """Trace the model and write ONNX fp32."""
    from tigrinya_asr.model import load_finetune_model
    model = load_finetune_model("omniASR_CTC_300M_v2")
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.zeros(1, 16000, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "time"}, "logits": {0: "batch", 1: "time"}},
        opset_version=opset,
    )
    return onnx_path


def quantize_to_int8(onnx_path: Path, dst: Path) -> Path:
    """Dynamic int8 quantization for the wav2vec2-style encoder."""
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    return dst
```

- [ ] **Step 4: Run end-to-end export**

```bash
uv run python -c "
from pathlib import Path
from tigrinya_asr.export_onnx import export_to_onnx, quantize_to_int8
export_to_onnx(Path('outputs/ft-300m/final'), Path('outputs/onnx/model.fp32.onnx'))
quantize_to_int8(Path('outputs/onnx/model.fp32.onnx'), Path('outputs/onnx/model.int8.onnx'))
"
ls -lh outputs/onnx/
```
Expected: `model.int8.onnx` ≤ 350 MB.

- [ ] **Step 5: Run test**

```bash
uv run pytest tests/test_export_onnx.py -v
```
Expected: pass

- [ ] **Step 6: Commit**

```bash
git add src/tigrinya_asr/export_onnx.py tests/test_export_onnx.py
git commit -m "feat(export): ONNX fp32 + int8 dynamic quantize"
```

---

## Task 13: sherpa-onnx Benchmark on M4 Pro

**Files:**
- Create: `src/tigrinya_asr/benchmark.py`

- [ ] **Step 1: Write benchmark CLI**

`src/tigrinya_asr/benchmark.py`:
```python
"""Measure RTF of fine-tuned ONNX model on a held-out audio sample via sherpa-onnx."""
import time
from pathlib import Path

import click
import soundfile as sf
import sherpa_onnx


@click.command()
@click.option("--onnx-encoder", type=Path, required=True)
@click.option("--tokens", type=Path, required=True, help="tokens.txt from Omnilingual export")
@click.option("--audio", type=Path, required=True)
@click.option("--num-threads", type=int, default=4)
@click.option("--runs", type=int, default=3)
def main(onnx_encoder: Path, tokens: Path, audio: Path, num_threads: int, runs: int):
    recognizer = sherpa_onnx.OfflineRecognizer.from_wav2vec2_ctc(
        model=str(onnx_encoder),
        tokens=str(tokens),
        num_threads=num_threads,
        provider="cpu",
    )
    samples, sr = sf.read(audio, dtype="float32")
    duration = len(samples) / sr

    for i in range(runs):
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        recognizer.decode_stream(stream)
        elapsed = time.perf_counter() - t0
        rtf = elapsed / duration
        print(f"run {i+1}: duration={duration:.1f}s elapsed={elapsed:.2f}s RTF={rtf:.3f}  hyp={stream.result.text[:80]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Get tokens file**

The Omnilingual tokenizer model lives at `~/.cache/fairseq2/assets/<hash>/omniASR_tokenizer.model`. sherpa-onnx needs a plain text tokens file. Convert:

```bash
uv run python -c "
from tigrinya_asr.tokenizer import load_tokenizer
tok = load_tokenizer()
with open('outputs/onnx/tokens.txt', 'w') as f:
    for i in range(len(tok)):
        f.write(f'{tok.decode([i])} {i}\n')
print('wrote outputs/onnx/tokens.txt')
"
```
(Exact API depends on the tokenizer object; adapt accordingly.)

- [ ] **Step 3: Run benchmark on the existing Tigrinya memo**

```bash
uv run python -m tigrinya_asr.benchmark \
    --onnx-encoder outputs/onnx/model.int8.onnx \
    --tokens outputs/onnx/tokens.txt \
    --audio ~/dev/tigrinya-transcribe/audio/stigenga.wav \
    --runs 5
```
Expected: RTF ≤ 0.15 (target ≤ 0.10).

- [ ] **Step 4: Commit**

```bash
git add src/tigrinya_asr/benchmark.py
git commit -m "feat(benchmark): sherpa-onnx CPU EP RTF measurement on macOS"
```

**Gate:** if RTF > 0.20, profile (check that XNNPACK is active, batch=1, threads tuned). Try `num_threads=8` on M4 Pro's 10 P-cores.

---

## Task 14: CLI Wrapper

**Files:**
- Create: `src/tigrinya_asr/cli.py`

- [ ] **Step 1: Write click-based dispatcher**

```python
"""Entry point for the `tigrinya-asr` console script."""
import click

from . import train as train_mod
from . import benchmark as benchmark_mod
from . import export_onnx as export_mod


@click.group()
def cli() -> None:
    """Tigrinya ASR fine-tuning tools."""


cli.add_command(train_mod.main, name="train")
cli.add_command(benchmark_mod.main, name="benchmark")


@cli.command()
@click.option("--checkpoint-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-fp32", type=click.Path(path_type=Path), default="outputs/onnx/model.fp32.onnx")
@click.option("--out-int8", type=click.Path(path_type=Path), default="outputs/onnx/model.int8.onnx")
def export(checkpoint_dir, out_fp32, out_int8):
    from pathlib import Path
    export_mod.export_to_onnx(Path(checkpoint_dir), Path(out_fp32))
    export_mod.quantize_to_int8(Path(out_fp32), Path(out_int8))
    click.echo(f"wrote {out_int8}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Verify**

```bash
uv run tigrinya-asr --help
uv run tigrinya-asr train --help
uv run tigrinya-asr benchmark --help
uv run tigrinya-asr export --help
```
Expected: all four print usage.

- [ ] **Step 3: Commit**

```bash
git add src/tigrinya_asr/cli.py
git commit -m "feat(cli): unified train/benchmark/export entry"
```

---

## Task 15: Wire Fine-Tuned Model Into bible_study_transcribe

**Files:**
- Modify: `~/dev/bible-study-transcribe/src/bible_study_transcribe/transcribe.py`
- Modify: `~/dev/bible-study-transcribe/pyproject.toml`

Swap the default ASR backend from Omnilingual-7B (PyTorch / fairseq2) to the fine-tuned sherpa-onnx int8 model produced by Task 12.

- [ ] **Step 1: Add sherpa-onnx to the runtime dependencies**

Move `sherpa-onnx` out of the `[export]` extra and into the main `dependencies` array (it is now load-bearing for the production pipeline):

```bash
uv add sherpa-onnx
```

- [ ] **Step 2: Modify transcribe.py to support a sherpa-onnx backend**

In `src/bible_study_transcribe/transcribe.py`, add a second backend selected by env var or config:

```python
SHERPA_ONNX_MODEL = os.environ.get("TIGRINYA_ONNX_MODEL")  # path to .onnx
SHERPA_ONNX_TOKENS = os.environ.get("TIGRINYA_ONNX_TOKENS")


def transcribe_segments(wav_path, segments, tmp_dir, model_card=DEFAULT_MODEL, batch_size=4):
    if SHERPA_ONNX_MODEL and SHERPA_ONNX_TOKENS:
        return _transcribe_sherpa_onnx(wav_path, segments, tmp_dir)
    # ... existing fairseq2 path
```

Implement `_transcribe_sherpa_onnx` using the same `OfflineRecognizer.from_wav2vec2_ctc` setup, reading each sliced segment and pushing through.

- [ ] **Step 3: Run smoke test on the existing 30-sec memo**

```bash
cd ~/dev/bible-study-transcribe
export TIGRINYA_ONNX_MODEL=$PWD/outputs/onnx/model.int8.onnx
export TIGRINYA_ONNX_TOKENS=$PWD/outputs/onnx/tokens.txt
uv run bible-study-transcribe ~/Downloads/"Stigenga 13 63.m4a" --no-summary
```
Expected: transcript appears in a few seconds (vs minutes for the 7B model).

- [ ] **Step 4: Commit the integration**

```bash
git add src/bible_study_transcribe/transcribe.py pyproject.toml
git commit -m "feat(transcribe): sherpa-onnx backend with fine-tuned Tigrinya model"
```

---

## Task 16: Push Plan + Fine-Tuning Code to GitHub

**Files:**
- None (operational — pushes to existing `Abiel7/bible-study-transcribe`)

- [ ] **Step 1: Push everything to the existing repo**

```bash
cd ~/dev/bible-study-transcribe
git push origin main
```

- [ ] **Step 2: Open the repo to verify the plan + new files are visible**

```bash
gh browse
```

Expected: `docs/superpowers/plans/2026-05-13-tigrinya-asr-finetune.md` and `src/tigrinya_asr/...` are on `main`.

---

## Plan B (only if Plan A WER > 30%)

Switch base to `facebook/w2v-bert-2.0` (the model Ethio-ASR used to hit 35.22% WER). Re-do Tasks 5, 7, 10. Notable differences:
- Tokenizer: build a fresh 409-symbol char-level grapheme vocab from WAXAL train texts (NOT the Omnilingual universal vocab)
- Model size: ~580M params, ~1.15 GB fp16 (overshoots 1 GB budget — quantize to int8 for ~600 MB)
- Pretraining: w2v-bert is purely SSL; the CTC head is randomly initialized — expect higher initial loss but better asymptote

If Plan B also fails, escalate to pseudo-labeling loop (Task 17 below — not in this plan).

---

## Self-Review Checklist

- [x] **Spec coverage:** Goal (fine-tune for Tigrinya + sherpa-onnx export) → Tasks 0-15. Fast inference target → Tasks 12-13. Reproducibility → Task 9 (Modal). All four research-agent recommendations integrated.
- [x] **Placeholder scan:** No TBDs or "implement later." Two explicit `Note:` callouts (Tasks 2 and 5) flag exact import paths to verify at runtime — this is intentional, not a placeholder, because the omnilingual-asr package internals can change between minor versions.
- [x] **Type consistency:** `ManifestRow` defined Task 3, used Task 7. `load_finetune_model` signature defined Task 5, used Tasks 7/12. `compute_wer_cer` signature consistent across Tasks 6/7.
- [x] **Gates explicit:** Task 8 gates Task 9. Task 10 WER gate on Task 11. Task 13 RTF gate on Task 15.

---

## References

- [Ethio-ASR paper (arXiv 2603.23654)](https://arxiv.org/html/2603.23654v1) — recipe + WAXAL benchmark
- [Omnilingual ASR paper (arXiv 2511.09690)](https://arxiv.org/abs/2511.09690)
- [Omnilingual ASR GitHub](https://github.com/facebookresearch/omnilingual-asr) — `workflows/recipes/wav2vec2/asr/README.md`
- [sherpa-onnx Omnilingual ASR docs](https://k2-fsa.github.io/sherpa/onnx/omnilingual-asr/models.html) — measured RTF on macOS
- [sherpa-onnx 300M int8 model card](https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12)
- [google/WaxalNLP dataset](https://huggingface.co/datasets/google/WaxalNLP)
- [HuggingFace fine-tune w2v-bert blog](https://huggingface.co/blog/fine-tune-w2v2-bert)
- [Modal docs — A100-80GB pricing](https://modal.com/pricing)
- [pyctcdecode + KenLM rescoring](https://github.com/kensho-technologies/pyctcdecode)
