# transcriber

Пайплайн автоматической транскрипции и диаризации аудио/видео файлов для Apple Silicon.

Кладёт файлы в `input/`, запускает скрипт — получает Markdown с таймкодами и именами спикеров в `output/`.

## Скорость

| | Время | RTF |
|---|---|---|
| Предыдущий (mlx-whisper + pyannote, CPU) | 10:13 | 2.7× медленнее |
| **Текущий (Groq API + simple-diarizer)** | **0:36** | **6.4× быстрее** реального времени |

Тест: файл 3:49, Apple M2.

## Как работает

1. Любой входной файл конвертируется в WAV 16 kHz mono через `ffmpeg`
2. Параллельно запускаются два процесса:
   - **Транскрипция** — Groq Whisper API (~секунды); при отсутствии ключа **или любой ошибке API** автоматически переключается на mlx-whisper локально (Metal GPU/Neural Engine, ~10–15 мин для часа записи)
   - **Диаризация** — simple-diarizer ECAPA-TDNN (CPU); fallback: pyannote.audio
3. Сегменты Whisper совмещаются с метками спикеров по таймкодам
4. Сохраняется `output/<имя>.md`

## Пример вывода

```markdown
# Транскрипция: meeting.mp4

- Модель ASR: Whisper `groq/whisper-large-v3-turbo`
- Диаризация: `simple-diarizer/ecapa`

**Спикер 1 [00:00:03–00:00:10]:** Добрый день, начнём совещание.

**Спикер 2 [00:00:11–00:00:18]:** Да, всем привет.
```

## Режимы работы

| | Онлайн (быстро) | Офлайн (локально) |
|---|---|---|
| **Транскрипция** | Groq Whisper API | mlx-whisper (Metal GPU) |
| **Скорость** | ~30 сек на час записи | ~10–15 мин на час записи |
| **Что нужно** | `groq_key.txt` с ключом | ничего дополнительно |
| **Переключение** | автоматически, если нет ключа или API упал | — |

Диаризация всегда работает локально: simple-diarizer (ECAPA-TDNN) → pyannote.audio как fallback.

## Требования

- Python 3.10+
- `ffmpeg` (должен быть доступен в `PATH`)
- **Groq API ключ** (опционально, бесплатно, 2 ч/день): [console.groq.com](https://console.groq.com) → API Keys

HuggingFace токен нужен только для fallback-диаризации через pyannote (если simple-diarizer не установлен).

## Установка

```bash
python -m venv .venv
source .venv/bin/activate

pip install groq simple-diarizer mlx-whisper pyannote.audio torch
```

## Ключ Groq

```bash
echo "gsk_xxxxxxxxxx" > groq_key.txt
```

Либо через переменную окружения: `export GROQ_API_KEY=gsk_xxxxxxxxxx`

Без ключа скрипт автоматически переключается на mlx-whisper (Metal GPU/Neural Engine).

## Использование

```bash
cp /path/to/meeting.mp4 input/
source .venv/bin/activate
python transcribe_diarize.py
```

Файлы, для которых транскрипт уже существует в `output/`, пропускаются.

## Поддерживаемые форматы

| Тип | Расширения |
|---|---|
| Аудио | `.wav` `.mp3` `.m4a` `.flac` `.aac` `.ogg` |
| Видео | `.mp4` `.mov` `.mkv` `.avi` `.m4v` `.webm` |

## Конфигурация

В начале `transcribe_diarize.py`:

```python
MLX_MODEL_REPO     = "mlx-community/whisper-large-v3-turbo"  # fallback-модель
GROQ_MODEL         = "whisper-large-v3-turbo"                 # Groq-модель
GROQ_MAX_FILE_BYTES = 24 * 1024 * 1024  # порог разбивки на чанки (24 МБ)
GROQ_CHUNK_SEC     = 600                # длина чанка в секундах (10 мин)
```

### Большие файлы

Groq API ограничивает загружаемый файл 25 МБ. WAV 16kHz mono весит ~1.8 МБ/мин, поэтому файлы длиннее ~13 минут автоматически разбиваются на чанки по 10 минут. Чанки транскрибируются последовательно, таймкоды корректируются, временные файлы удаляются после каждого чанка.

## Структура проекта

```
transcriber/
├── transcribe_diarize.py   # основной скрипт
├── groq_key.txt            # ключ Groq API (не коммитить!)
├── hf_token.txt            # токен HuggingFace, нужен только для pyannote fallback
├── input/                  # входные файлы
├── output/                 # готовые транскрипты (.md)
└── temp/                   # временные WAV-файлы (удаляются автоматически)
```
