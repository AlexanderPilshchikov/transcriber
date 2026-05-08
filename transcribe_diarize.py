#!/usr/bin/env python3
"""
Пайплайн транскрипции и диаризации аудио/видео:
- ffmpeg  → WAV 16 kHz mono
- Whisper → транскрипция (Groq API или mlx-whisper как fallback)
- simple-diarizer → диаризация (pyannote как fallback)
- Мёрдж по таймкодам → Markdown в output/

Транскрипция и диаризация запускаются параллельно через ThreadPoolExecutor.
"""

import concurrent.futures
import os
import subprocess
from pathlib import Path

import mlx_whisper
import torch
from groq import Groq
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


# ------------ Конфигурация ------------

# HuggingFace репо с MLX-весами (fallback если нет Groq-ключа)
MLX_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
# Модель Groq Whisper (основной бэкенд)
GROQ_MODEL = "whisper-large-v3-turbo"
# Groq ограничение: 25 МБ. Берём 24 МБ с запасом.
GROQ_MAX_FILE_BYTES = 24 * 1024 * 1024
# Длина чанка при разбивке большого файла (10 минут)
GROQ_CHUNK_SEC = 600

# Папки проекта (создаются автоматически рядом со скриптом)
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
TEMP_DIR_NAME = "temp"

# Расширения файлов
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


# ------------ Вспомогательные функции ------------

def get_base_dirs():
    """Возвращает пути к base/input/output/temp и создаёт папки при необходимости."""
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / INPUT_DIR_NAME
    output_dir = base_dir / OUTPUT_DIR_NAME
    temp_dir = base_dir / TEMP_DIR_NAME
    for d in (input_dir, output_dir, temp_dir):
        d.mkdir(exist_ok=True)
    return base_dir, input_dir, output_dir, temp_dir


def detect_device():
    """Определяем, на чём считать: CUDA, MPS (Apple Silicon) или CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def convert_to_mono16k_wav(input_path: Path, temp_dir: Path) -> Path:
    """
    Приводит любой входной файл (аудио или видео) к WAV 16 kHz mono.
    Используется и для .webm/.mp4, и для .m4a/.mp3/...
    """
    audio_path = temp_dir / f"{input_path.stem}_16k.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",                # убрать видео, если есть
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    print(f"[ffmpeg] Конвертирую в WAV 16kHz mono: {input_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def get_audio_duration(audio_path: Path) -> float:
    """
    Возвращает длительность аудио в секундах через ffprobe.

    ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 input.wav
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(out.strip())
    except Exception as e:
        print(f"[warn] Не удалось получить длительность аудио {audio_path.name}: {e}")
        return 0.0


def format_time(seconds: float) -> str:
    """Форматирует секунды в вид HH:MM:SS."""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_hf_token(base_dir: Path) -> str:
    """
    Возвращает токен Hugging Face:
    1) сначала пробует переменную окружения HUGGINGFACE_TOKEN
    2) потом файл hf_token.txt в корне проекта
    """
    env_token = os.environ.get("HUGGINGFACE_TOKEN")
    if env_token:
        return env_token.strip()

    token_file = base_dir / "hf_token.txt"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise RuntimeError(
        "Не найден токен Hugging Face.\n"
        "Либо экспортируй HUGGINGFACE_TOKEN, либо создай файл hf_token.txt "
        "в корне проекта с токеном внутри одной строкой."
    )


def get_groq_key(base_dir: Path):
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    key_file = base_dir / "groq_key.txt"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


# ------------ Whisper и диаризация ------------

def split_audio(audio_path: Path, chunk_sec: int) -> list[Path]:
    """Разбивает WAV на чанки по chunk_sec секунд через ffmpeg segment muxer."""
    temp_dir = audio_path.parent
    chunk_pattern = temp_dir / f"{audio_path.stem}_chunk%03d.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_sec),
        "-c", "copy",
        str(chunk_pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(temp_dir.glob(f"{audio_path.stem}_chunk*.wav"))


def _transcribe_chunk(client, chunk_path: Path, offset: float) -> list[dict]:
    """Транскрибирует один чанк через Groq API, возвращает сегменты с поправкой на offset."""
    with open(chunk_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=GROQ_MODEL,
            file=(chunk_path.name, f),
            response_format="verbose_json",
        )
    return [
        {"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"]}
        for s in result.segments
    ]


def run_whisper(audio_path: Path, groq_key: str | None = None):
    if groq_key:
        file_size = audio_path.stat().st_size
        client = Groq(api_key=groq_key)

        if file_size > GROQ_MAX_FILE_BYTES:
            size_mb = file_size / 1024 / 1024
            print(
                f"[whisper] Файл {audio_path.name} ({size_mb:.0f} МБ) превышает "
                f"лимит Groq 25 МБ, разбиваю на чанки по {GROQ_CHUNK_SEC // 60} мин ..."
            )
            chunks = split_audio(audio_path, GROQ_CHUNK_SEC)
            print(f"[whisper] Чанков: {len(chunks)}")
            all_segments = []
            for i, chunk_path in enumerate(chunks):
                offset = i * GROQ_CHUNK_SEC
                print(
                    f"[whisper] Чанк {i + 1}/{len(chunks)}: {chunk_path.name} "
                    f"(смещение {format_time(offset)}) ..."
                )
                all_segments.extend(_transcribe_chunk(client, chunk_path, float(offset)))
                chunk_path.unlink()
            print(f"[whisper] Готово: {audio_path.name} ({len(all_segments)} сегментов)")
            return {"segments": all_segments}

        print(f"[whisper] Транскрибирую {audio_path.name} (Groq API) ...")
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=GROQ_MODEL,
                file=(audio_path.name, f),
                response_format="verbose_json",
            )
        print(f"[whisper] Готово: {audio_path.name}")
        return {"segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.segments
        ]}
    else:
        print(f"[whisper] Транскрибирую {audio_path.name} (MLX, GPU/Neural Engine) ...")
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=MLX_MODEL_REPO,
            verbose=False,
        )
        print(f"[whisper] Готово: {audio_path.name}")
        return result


def run_simple_diarizer(audio_path: Path):
    # silero-vad грузится через torch.hub и требует интерактивного подтверждения доверия;
    # добавляем его в trusted_list заранее, чтобы не блокироваться без терминала
    hub_dir = Path(torch.hub.get_dir())
    hub_dir.mkdir(parents=True, exist_ok=True)
    trusted_file = hub_dir / "trusted_list"
    if not trusted_file.exists() or "snakers4_silero-vad" not in trusted_file.read_text():
        with trusted_file.open("a") as f:
            f.write("snakers4_silero-vad\n")

    from simple_diarizer.diarizer import Diarizer
    print(f"[diarizer] Диаризация {audio_path.name} (simple-diarizer, ECAPA-TDNN) ...")
    diarizer = Diarizer(embed_model='ecapa', cluster_method='sc')
    segments = diarizer.diarize(str(audio_path), num_speakers=None, threshold=0.8)
    print(f"[diarizer] Готово: {audio_path.name}")
    return [
        {"start": s["start"], "end": s["end"], "speaker": f"SPEAKER_{s['label']:02d}"}
        for s in segments
    ]


def run_diarization(audio_path: Path, diar_pipeline=None):
    try:
        return run_simple_diarizer(audio_path)
    except ImportError:
        print("[diarizer] simple-diarizer не найден, использую pyannote")
        return run_pyannote(diar_pipeline, audio_path)


def run_pyannote(pipeline, audio_path: Path):
    """
    Запускает pyannote speaker diarization.

    В новых версиях pipeline(...) возвращает DiarizeOutput с полями:
    - speaker_diarization: Annotation (обычная диаризация)
    - exclusive_speaker_diarization: Annotation (без наложения)
    Нам нужен Annotation, у которого есть .itertracks()[web:61][web:36][web:53]
    """
    print(f"[pyannote] Диаризация {audio_path.name} (0% -> 100%) ...")
    with ProgressHook() as hook:
        diarization = pipeline(str(audio_path), hook=hook)
    print(f"[pyannote] Готово: {audio_path.name}")

    # Достаём Annotation из DiarizeOutput
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, "exclusive_speaker_diarization"):
        annotation = diarization.exclusive_speaker_diarization
    else:
        # Старый API: pipeline сразу возвращал Annotation
        annotation = diarization

    diar_segments = []
    if hasattr(annotation, "itertracks"):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            diar_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(speaker),
                }
            )
    else:
        print(
            f"[warn] annotation не поддерживает itertracks. "
            f"Тип: {type(annotation)}, атрибуты: "
            f"{[a for a in dir(annotation) if not a.startswith('__')]}"
        )

    if not diar_segments:
        print(
            f"[warn] pyannote не вернул ни одного сегмента. "
            f"Тип результата: {type(diarization)}"
        )

    return diar_segments


def assign_speakers_to_segments(whisper_segments, diar_segments):
    """
    Назначает каждому сегменту Whisper спикера на основе временных интервалов pyannote.

    Стратегия:
    - берём середину сегмента Whisper;
    - ищем интервал diarization, который её покрывает;
    - если не нашли, ищем максимальное пересечение.
    """
    for seg in whisper_segments:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", 0.0))
        mid = 0.5 * (s_start + s_end)

        best_speaker = None
        best_overlap = 0.0

        # 1. По попаданию середины в интервал
        for d in diar_segments:
            if d["start"] <= mid <= d["end"]:
                best_speaker = d["speaker"]
                break

        # 2. Если не попали ни в один интервал — по максимальному пересечению
        if best_speaker is None:
            for d in diar_segments:
                overlap = min(s_end, d["end"]) - max(s_start, d["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d["speaker"]

        seg["speaker"] = best_speaker or "UNKNOWN"

    return whisper_segments


def normalize_speaker_labels(segments):
    """
    Преобразует внутренние метки (speaker_0, SPEAKER_00, UNKNOWN)
    в человекочитаемые 'Спикер 1', 'Спикер 2', ... и записывает
    в поле 'speaker_readable'.
    """
    mapping = {}
    next_id = 1
    for seg in segments:
        raw = seg.get("speaker") or "UNKNOWN"
        if raw not in mapping:
            mapping[raw] = f"Спикер {next_id}"
            next_id += 1
        seg["speaker_readable"] = mapping[raw]
    return mapping


def build_markdown(source_file: Path, segments, model_name: str, diar_model_name: str) -> str:
    """
    Собирает Markdown-строку с транскрипцией по спикерам.
    """
    lines = []
    lines.append(f"# Транскрипция: {source_file.name}")
    lines.append("")
    lines.append(f"- Модель ASR: Whisper `{model_name}`")
    lines.append(f"- Диаризация: `{diar_model_name}`")
    lines.append("")

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker_readable", "Спикер ?")
        start = format_time(seg.get("start", 0.0))
        end = format_time(seg.get("end", 0.0))
        lines.append(f"**{speaker} [{start}–{end}]:** {text}")
        lines.append("")

    return "\n".join(lines)


# ------------ Основная обработка файла ------------

def process_file(
    file_path: Path,
    output_dir: Path,
    temp_dir: Path,
    diar_pipeline,
    diar_model_name: str,
    groq_key: str | None = None,
    asr_model_name: str = "",
):
    """
    Обрабатывает один файл: извлечение аудио (если нужно),
    транскрипция, диаризация, назначение спикеров, сохранение .md.
    """
    output_path = output_dir / f"{file_path.stem}.md"
    if output_path.exists():
        print(f"[skip] Уже есть транскрипт для {file_path.name}: {output_path.name}")
        return

    print(f"\n=== Обработка файла: {file_path.name} ===")

    # 1. Конвертация в WAV 16 kHz mono
    if is_audio_file(file_path):
        print("[stage] 1/4 Конвертация аудио → WAV 16kHz mono")
    elif is_video_file(file_path):
        print("[stage] 1/4 Извлечение аудио из видео → WAV 16kHz mono")
    else:
        print(f"[warn] Неизвестный тип файла, пропускаю: {file_path.name}")
        return
    audio_path = convert_to_mono16k_wav(file_path, temp_dir)

    total_audio_sec = get_audio_duration(audio_path)
    if total_audio_sec > 0:
        print(
            f"[info] Длительность аудио: {format_time(total_audio_sec)} "
            f"({total_audio_sec:.1f} с)"
        )

    # 2+3. Транскрипция и диаризация параллельно
    print("[stage] 2+3/4 Транскрипция и диаризация параллельно")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_whisper = executor.submit(run_whisper, audio_path, groq_key)
        future_diar = executor.submit(run_diarization, audio_path, diar_pipeline)
        whisper_result = future_whisper.result()
        diar_segments = future_diar.result()

    whisper_segments = whisper_result.get("segments", [])
    if not whisper_segments:
        print(f"[warn] Whisper не вернул сегментов для {file_path.name}")
        return

    print("[stage] 3/4 Назначение спикеров по таймкодам")
    whisper_segments = assign_speakers_to_segments(whisper_segments, diar_segments)
    normalize_speaker_labels(whisper_segments)

    # 4. Формирование и сохранение Markdown
    print("[stage] 4/4 Сохранение Markdown")
    md_text = build_markdown(file_path, whisper_segments, asr_model_name or "whisper", diar_model_name)
    output_path.write_text(md_text, encoding="utf-8")
    print(f"[ok] Сохранил транскрипт: {output_path}")

    # Удаляем временный аудиофайл, если он был создан из видео
    if audio_path.exists() and audio_path.parent == temp_dir:
        audio_path.unlink()
        print("[cleanup] Временный аудиофайл удалён")


# ------------ main ------------

def main():
    base_dir, input_dir, output_dir, temp_dir = get_base_dirs()
    device = detect_device()
    print(f"[init] device: {device}")

    # --- Whisper бэкенд ---
    groq_key = get_groq_key(base_dir)
    if groq_key:
        asr_model_name = f"groq/{GROQ_MODEL}"
        print(f"[init] Whisper: Groq API (модель {GROQ_MODEL})")
    else:
        asr_model_name = f"mlx/{MLX_MODEL_REPO.split('/')[-1]}"
        print(f"[init] Whisper: MLX '{MLX_MODEL_REPO}' (нет groq_key.txt, fallback)")

    # --- Диаризация бэкенд ---
    try:
        from simple_diarizer.diarizer import Diarizer as _Diarizer  # noqa: F401
        diar_pipeline = None
        diar_model_name = "simple-diarizer/ecapa"
        print("[init] Диаризация: simple-diarizer (ECAPA-TDNN)")
    except ImportError:
        print("[init] simple-diarizer не найден, загружаю pyannote...")
        hf_token = get_hf_token(base_dir)
        diar_model_name = "pyannote/speaker-diarization-3.1"
        print(f"[init] Загружаю pyannote pipeline '{diar_model_name}'...")
        diar_pipeline = Pipeline.from_pretrained(diar_model_name, token=hf_token)
        diar_device = device if device == "cuda" else "cpu"
        diar_pipeline.to(torch.device(diar_device))
        print(f"[init] pyannote будет работать на: {diar_device}")

    files = sorted(input_dir.iterdir())
    if not files:
        print(f"[info] Папка {input_dir} пуста, положи туда аудио/видео файлы.")
        return

    for path in files:
        if path.is_dir():
            continue
        if not (is_audio_file(path) or is_video_file(path)):
            print(f"[skip] Не аудио/видео: {path.name}")
            continue

        process_file(
            path,
            output_dir=output_dir,
            temp_dir=temp_dir,
            diar_pipeline=diar_pipeline,
            diar_model_name=diar_model_name,
            groq_key=groq_key,
            asr_model_name=asr_model_name,
        )


if __name__ == "__main__":
    main()