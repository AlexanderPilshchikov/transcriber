# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the script

```bash
source .venv/bin/activate
python transcribe_diarize.py
```

Place audio/video files in `input/` before running. Existing transcripts in `output/` are skipped automatically.

## Architecture

Single-file pipeline in `transcribe_diarize.py`. Per-file processing has 4 stages:

1. **Audio prep** (`convert_to_mono16k_wav`) — every input file (audio or video) is converted to mono 16 kHz WAV in `temp/` via `ffmpeg`. The temp file is always deleted after processing.
2. **faster-whisper** (`run_whisper`) — uses `WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")`. The generator returned by `model.transcribe()` is eagerly consumed into a list of `{"start", "end", "text"}` dicts to match the rest of the pipeline.
3. **pyannote** (`run_pyannote`) — returns diarization segments with `start`, `end`, `speaker`. Handles both old API (returns `Annotation` directly) and new API (returns `DiarizeOutput` with `.speaker_diarization`).
4. **Merge + output** (`assign_speakers_to_segments` → `normalize_speaker_labels` → `build_markdown`) — assigns a speaker to each Whisper segment by midpoint lookup, falls back to max-overlap; maps internal labels (`SPEAKER_00`, etc.) to "Спикер 1", "Спикер 2", …; writes Markdown to `output/`.

## Key configuration

At the top of `transcribe_diarize.py`:

```python
MODEL_NAME = "turbo"   # used only in Markdown output header
OMP_NUM_THREADS = 8    # also sets torch.set_num_threads(8)
```

The actual faster-whisper model (`large-v3-turbo`) and compute settings (`device="cpu"`, `compute_type="int8"`) are hardcoded in `main()`. pyannote model is hardcoded: `pyannote/speaker-diarization-3.1`.

## Device selection

`detect_device()` returns `cuda` → `mps` → `cpu`. Both faster-whisper and pyannote run on CPU (faster-whisper does not support MPS; pyannote is also forced to CPU when MPS is detected). On Apple Silicon, `int8` quantization on CPU is still 2–4× faster than old `openai-whisper`.

## HuggingFace token

`get_hf_token()` checks `HUGGINGFACE_TOKEN` env var first, then `hf_token.txt` in the project root. The token must have access to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

## Dependencies

Managed in `.venv/`. No `requirements.txt` — install manually:

```bash
pip install faster-whisper pyannote.audio torch
```

`ffmpeg` must be available in `PATH`.
