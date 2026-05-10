#!/usr/bin/env python3
"""MCP server — обёртка над пайплайном транскрипции/диаризации."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("transcriber")

# Состояние бэкендов инициализируется лениво при первом вызове
_ready = False
_groq_key = None
_diar_pipeline = None
_diar_model_name = None
_asr_model_name = None
_output_dir = None
_temp_dir = None


def _init():
    global _ready, _groq_key, _diar_pipeline, _diar_model_name
    global _asr_model_name, _output_dir, _temp_dir

    if _ready:
        return

    import transcribe_diarize as td

    base_dir, _, _output_dir, _temp_dir = td.get_base_dirs()
    _groq_key = td.get_groq_key(base_dir)
    _asr_model_name = (
        f"groq/{td.GROQ_MODEL}" if _groq_key
        else f"mlx/{td.MLX_MODEL_REPO.split('/')[-1]}"
    )

    try:
        from simple_diarizer.diarizer import Diarizer  # noqa: F401
        _diar_pipeline = None
        _diar_model_name = "simple-diarizer/ecapa"
    except ImportError:
        import torch
        from pyannote.audio import Pipeline
        hf_token = td.get_hf_token(base_dir)
        _diar_model_name = "pyannote/speaker-diarization-3.1"
        _diar_pipeline = Pipeline.from_pretrained(_diar_model_name, token=hf_token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _diar_pipeline.to(torch.device(device))

    _ready = True


@mcp.tool()
def transcribe_file(file_path: str) -> str:
    """Транскрибирует аудио или видео файл с диаризацией спикеров.

    Принимает абсолютный путь к файлу (wav, mp3, m4a, flac, aac, ogg,
    mp4, mov, mkv, avi, m4v, webm). Возвращает Markdown с транскрипцией
    по спикерам и таймкодами.
    """
    import transcribe_diarize as td

    _init()

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return f"Ошибка: файл не найден: {path}"
    if not (td.is_audio_file(path) or td.is_video_file(path)):
        supported = td.AUDIO_EXTS | td.VIDEO_EXTS
        return f"Ошибка: неподдерживаемый формат '{path.suffix}'. Поддерживаются: {sorted(supported)}"

    output_path = _output_dir / f"{path.stem}.md"

    # Если транскрипт уже есть — сразу вернуть его
    if output_path.exists():
        return output_path.read_text(encoding="utf-8")

    td.process_file(
        path,
        output_dir=_output_dir,
        temp_dir=_temp_dir,
        diar_pipeline=_diar_pipeline,
        diar_model_name=_diar_model_name,
        groq_key=_groq_key,
        asr_model_name=_asr_model_name,
    )

    if output_path.exists():
        return output_path.read_text(encoding="utf-8")
    return f"Ошибка: транскрипт не создан для {path.name}"


@mcp.tool()
def list_transcripts() -> str:
    """Возвращает список уже готовых транскриптов в output/."""
    _init()
    files = sorted(_output_dir.glob("*.md"))
    if not files:
        return "Транскриптов пока нет."
    return "\n".join(f"- {f.name}" for f in files)


@mcp.tool()
def get_transcript(stem: str) -> str:
    """Возвращает содержимое готового транскрипта по имени файла без расширения.

    Например: get_transcript('meeting_2024-01-15')
    """
    _init()
    output_path = _output_dir / f"{stem}.md"
    if not output_path.exists():
        available = [f.stem for f in sorted(_output_dir.glob("*.md"))]
        return f"Транскрипт '{stem}' не найден. Доступные: {available}"
    return output_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run()
