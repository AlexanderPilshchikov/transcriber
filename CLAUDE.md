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

1. **Audio prep** (`convert_to_mono16k_wav`) — every input file (audio or video) is converted to mono 16 kHz WAV in `temp/` via `ffmpeg`. Temp file is deleted after processing.
2. **Transcription + Diarization in parallel** (`ThreadPoolExecutor(max_workers=2)`):
   - `run_whisper(audio_path, groq_key)` — if `groq_key` is set, calls Groq Whisper API (network I/O, ~seconds); otherwise falls back to `mlx_whisper.transcribe()` on Metal GPU/Neural Engine. Returns `{"segments": [{"start", "end", "text"}]}`.
   - `run_diarization(audio_path, diar_pipeline)` — tries `run_simple_diarizer()` first (ECAPA-TDNN via speechbrain, CPU); falls back to `run_pyannote()` on `ImportError`.
3. **Merge** (`assign_speakers_to_segments` → `normalize_speaker_labels`) — assigns speaker to each Whisper segment by midpoint lookup, falls back to max-overlap. Maps internal labels to "Спикер 1", "Спикер 2", …
4. **Output** (`build_markdown`) — writes `output/<stem>.md`.

## Key configuration

```python
MLX_MODEL_REPO      = "mlx-community/whisper-large-v3-turbo"  # fallback when no Groq key
GROQ_MODEL          = "whisper-large-v3-turbo"                 # Groq model name
GROQ_MAX_FILE_BYTES = 24 * 1024 * 1024  # Groq upload limit is 25 MB; 24 MB with margin
GROQ_CHUNK_SEC      = 600               # chunk length when splitting (10 min)
```

## Large-file chunking (Groq)

`run_whisper()` checks the WAV size before uploading. If it exceeds `GROQ_MAX_FILE_BYTES`:

1. `split_audio()` calls `ffmpeg -f segment` to cut the WAV into `GROQ_CHUNK_SEC`-second chunks stored in `temp/`.
2. `_transcribe_chunk()` uploads each chunk and shifts every segment's `start`/`end` by `i * GROQ_CHUNK_SEC`.
3. Chunk files are deleted immediately after transcription.

WAV 16 kHz mono is ~1.8 MB/min, so files longer than ~13 min trigger chunking.

## Backend selection (runtime)

`main()` auto-selects backends:

| Backend | Condition |
|---|---|
| Groq API (transcription) | `groq_key.txt` exists or `GROQ_API_KEY` env var set |
| mlx-whisper fallback | No Groq key found |
| simple-diarizer (diarization) | `simple_diarizer` importable |
| pyannote fallback | `ImportError` on simple_diarizer import |

pyannote requires HuggingFace token (`hf_token.txt` or `HUGGINGFACE_TOKEN` env var). simple-diarizer does not.

## silero-vad trust

`run_simple_diarizer()` pre-adds `snakers4_silero-vad` to `~/.cache/torch/hub/trusted_list` before loading the Diarizer to avoid an interactive stdin prompt when running without a terminal.

## Dependencies

Managed in `.venv/`. No `requirements.txt` — install manually:

```bash
pip install groq simple-diarizer mlx-whisper pyannote.audio torch
```

`ffmpeg` must be available in `PATH`.
