# transcriber

Пайплайн автоматической транскрипции и диаризации аудио/видео файлов.

Скрипт берёт файлы из папки `input/`, транскрибирует их через **faster-whisper** (`large-v3-turbo`), определяет спикеров через **pyannote.audio** и сохраняет результат в `output/` в виде Markdown с таймкодами.

## Как работает

1. Сканирует папку `input/` на наличие аудио/видео файлов
2. Любой входной файл конвертирует в WAV 16 kHz mono через `ffmpeg` (и аудио, и видео)
3. Транскрибирует аудио — faster-whisper возвращает сегменты с таймкодами
4. Диаризирует — pyannote определяет, кто и когда говорил
5. Совмещает транскрипцию с диаризацией по временным интервалам
6. Сохраняет `output/<имя_файла>.md` с разметкой по спикерам

## Пример вывода

```markdown
# Транскрипция: meeting.mp4

- Модель ASR: Whisper `turbo`
- Диаризация: `pyannote/speaker-diarization-3.1`

**Спикер 1 [00:00:03–00:00:10]:** Добрый день, начнём совещание.

**Спикер 2 [00:00:11–00:00:18]:** Да, всем привет.
```

## Требования

- Python 3.9+
- `ffmpeg` (должен быть доступен в `PATH`)
- Токен [Hugging Face](https://huggingface.co/settings/tokens) с принятыми условиями моделей:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Python-зависимости

| Пакет | Версия |
|---|---|
| faster-whisper | — |
| pyannote.audio | 4.0.4 |
| torch | 2.11.0 |

## Установка

```bash
# Клонировать / скачать репозиторий
cd transcriber

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Установить зависимости
pip install faster-whisper pyannote.audio torch
```

## Токен Hugging Face

Создать файл `hf_token.txt` в корне проекта и вставить туда токен одной строкой:

```
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Либо экспортировать переменную окружения:

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Использование

```bash
# Положить файлы в input/
cp /path/to/meeting.mp4 input/

# Запустить
source .venv/bin/activate
python transcribe_diarize.py
```

Готовые транскрипты появятся в `output/`. Файлы, для которых транскрипт уже существует, повторно не обрабатываются.

## Поддерживаемые форматы

| Тип | Расширения |
|---|---|
| Аудио | `.wav` `.mp3` `.m4a` `.flac` `.aac` `.ogg` |
| Видео | `.mp4` `.mov` `.mkv` `.avi` `.m4v` `.webm` |

## Конфигурация

В начале файла `transcribe_diarize.py` можно изменить:

```python
MODEL_NAME = "turbo"   # tiny / base / small / medium / large / large-v3-turbo
```

Фактически используется `large-v3-turbo` с `int8`-квантизацией на CPU — на Apple Silicon M1/M2/M3 работает в 2–4 раза быстрее старого `openai-whisper`.

## Железо

Скрипт автоматически выбирает устройство:

| Устройство | Условие |
|---|---|
| CUDA | NVIDIA GPU |
| MPS | Apple Silicon (M1/M2/M3) |
| CPU | fallback |

> **Примечание:** faster-whisper и pyannote работают на CPU (MPS не поддерживается). На Apple Silicon всё выполняется на CPU с `int8`-квантизацией — это всё равно существенно быстрее старого `openai-whisper`.

## Структура проекта

```
transcriber/
├── transcribe_diarize.py   # основной скрипт
├── hf_token.txt            # токен HuggingFace (не коммитить!)
├── input/                  # входные файлы
├── output/                 # готовые транскрипты (.md)
└── temp/                   # временные WAV-файлы (создаются для каждого входного файла)
```
