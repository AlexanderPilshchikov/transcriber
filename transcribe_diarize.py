#!/usr/bin/env python3
"""
Пайплайн:
- Берём все аудио/видео из папки input/
- Для видео вытаскиваем аудио через ffmpeg
- Whisper -> транскрипция с сегментами
- pyannote.audio -> диаризация
- Мерджим по таймкодам, сохраняем Markdown в output/
"""

import os
import subprocess
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


# ------------ Конфигурация ------------

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# Модель Whisper: tiny/base/small/medium/large
MODEL_NAME = "turbo"

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


# ------------ Whisper и pyannote ------------

def run_whisper(model, audio_path: Path):
    print(f"[whisper] Транскрибирую {audio_path.name} ...")
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    
    # Приводим вывод faster-whisper к формату, который ждет наш старый код (список словарей)
    whisper_segments = []
    for segment in segments:
        whisper_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    print(f"[whisper] Готово: {audio_path.name}")
    return {"segments": whisper_segments}


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
    whisper_model,
    diar_pipeline,
    diar_model_name: str,
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

    # 1. Подготовка/извлечение аудио
    if is_audio_file(file_path):
        print("[stage] 1/4 Нормализация аудио в WAV 16kHz mono через ffmpeg")
        audio_path = convert_to_mono16k_wav(file_path, temp_dir)
    elif is_video_file(file_path):
        print("[stage] 1/4 Извлечение и нормализация аудио из видео через ffmpeg")
        audio_path = convert_to_mono16k_wav(file_path, temp_dir)
    else:
        print(f"[warn] Неизвестный тип файла, пропускаю: {file_path.name}")
        return

    total_audio_sec = get_audio_duration(audio_path)
    if total_audio_sec > 0:
        print(
            f"[info] Длительность аудио: {format_time(total_audio_sec)} "
            f"({total_audio_sec:.1f} с)"
        )

    # 2. Транскрипция Whisper
    print("[stage] 2/4 Транскрипция (Whisper)")
    whisper_result = run_whisper(whisper_model, audio_path)
    whisper_segments = whisper_result.get("segments", [])
    if not whisper_segments:
        print(f"[warn] Whisper не вернул сегментов для {file_path.name}")
        return

    # 3. Диаризация pyannote
    print("[stage] 3/4 Диаризация (pyannote)")
    diar_segments = run_pyannote(diar_pipeline, audio_path)

    print("[stage] 3.5/4 Назначение спикеров по таймкодам")
    whisper_segments = assign_speakers_to_segments(whisper_segments, diar_segments)
    normalize_speaker_labels(whisper_segments)

    # 4. Формирование и сохранение Markdown
    print("[stage] 4/4 Формирование и сохранение Markdown")
    md_text = build_markdown(file_path, whisper_segments, MODEL_NAME, diar_model_name)
    output_path.write_text(md_text, encoding="utf-8")
    print(f"[ok] Сохранил транскрипт: {output_path}")

    # Удаляем временный аудиофайл, если он был создан из видео
    if audio_path.exists() and audio_path.parent == temp_dir:
        audio_path.unlink()
        print("[cleanup] Временный аудиофайл удалён")


# ------------ main ------------

def main():
    base_dir, input_dir, output_dir, temp_dir = get_base_dirs()

    hf_token = get_hf_token(base_dir)

    device = detect_device()
    print(f"[init] device: {device}")

    print(f"[init] Загружаю Whisper модель '{MODEL_NAME}'...")
    # faster-whisper использует "cpu" или "cuda" (и "auto"). Для Mac (MPS) пока лучше использовать CPU + int8,
    # на M2 это всё равно работает в 2-4 раза быстрее старого Whisper
    whisper_model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8") 

    diar_model_name = "pyannote/speaker-diarization-3.1"
    print(f"[init] Загружаю pyannote pipeline '{diar_model_name}'...")
    diar_pipeline = Pipeline.from_pretrained(
        diar_model_name,
        token=hf_token,
    )
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
            whisper_model=whisper_model,
            diar_pipeline=diar_pipeline,
            diar_model_name=diar_model_name,
        )


if __name__ == "__main__":
    main()