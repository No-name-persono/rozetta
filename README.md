# Rozetta

Rozetta — небольшой сервис для обработки и оценки звонков. Он принимает аудиозаписи, отправляет их в Yandex SpeechKit STT, при необходимости строит мягкие метки спикеров, передаёт расшифровку в LLM и возвращает структурированный JSON для CRM, BI, внутренней админки или другого внешнего хранилища.

Проект лучше воспринимать не как архив результатов, а как обработчик: загрузили запись, получили расшифровку, чек-лист, бизнес-статус и цитаты с таймкодами, сохранили результат у себя.

## Что умеет

- пакетная обработка аудиофайлов;
- SpeechKit STT v3 по умолчанию;
- анализ звонка через YandexGPT-совместимый API;
- редактируемые промты и чек-листы;
- отдельный API с Bearer-ключами;
- CLI для создания и отзыва API-ключей;
- мягкие spectral-метки спикеров после STT, по речевым диапазонам из word timestamps;
- Docker/Docker Compose запуск.

## Быстрый запуск через Docker

Создайте файл окружения:

```bash
cp .env.example .env
```

Заполните в `.env` параметры Yandex Cloud, Object Storage и LLM, затем запустите:

```bash
docker compose up --build -d
```

Проверка здоровья сервиса:

```bash
curl http://127.0.0.1:5010/api/v1/health
```

Веб-интерфейс будет доступен на:

```text
http://127.0.0.1:5010/
```

## Настройки

Основные переменные в `.env`:

- `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` — S3/Object Storage для временной передачи аудио в SpeechKit;
- `YANDEX_API_KEY` или `YANDEX_IAM_TOKEN` — авторизация в SpeechKit;
- `YANDEX_FOLDER_ID` — folder id в Yandex Cloud;
- `ASR_API_VERSION=v3` — версия STT API;
- `ASR_MODEL=general` — модель распознавания;
- `LLM_MODEL_URI`, `LLM_API_KEY` или `LLM_IAM_TOKEN` — параметры LLM;
- `SPECTRAL_ENABLED=1` — включить мягкие метки спикеров;
- `SPECTRAL_STT_GUIDED_ENABLED=1` — строить мягкие метки только по участкам речи, найденным STT;
- `SPECTRAL_STT_MAX_GAP_SEC=0.8` — максимальная пауза между словами для склейки в один речевой диапазон;
- `SPECTRAL_STT_PADDING_SEC=0.2` — небольшой запас вокруг речевых диапазонов;
- `ASYNC_MAX_WORKERS=4` — параллельность обработки;
- `MAX_UPLOAD_MB=64` — максимальный размер загрузки.

## API-ключи

API-ключи создаются только из терминала и не отображаются в веб-интерфейсе. В SQLite хранится SHA-256 хэш ключа, а сам ключ показывается один раз при создании.

В Docker:

```bash
docker compose exec rozeeta python api_keys.py create my-integration
docker compose exec rozeeta python api_keys.py list
docker compose exec rozeeta python api_keys.py revoke 1
```

Локально:

```bash
python api_keys.py create my-integration
python api_keys.py list
python api_keys.py revoke 1
```

Сохраните созданный ключ сразу. Повторно показать его нельзя.

## API

Для закрытых методов используйте заголовок:

```http
Authorization: Bearer <api-key>
```

Открытая проверка здоровья:

```bash
curl http://127.0.0.1:5010/api/v1/health
```

Получить список промтов и чек-листов:

```bash
curl -H "Authorization: Bearer $ROZEETA_API_KEY" \
  http://127.0.0.1:5010/api/v1/templates
```

Создать batch:

```bash
curl -X POST http://127.0.0.1:5010/api/v1/batches \
  -H "Authorization: Bearer $ROZEETA_API_KEY" \
  -F "prompt_id=<prompt-id>" \
  -F "checklist_id=<checklist-id>" \
  -F "audio_files=@/path/to/call.mp3"
```

Проверить статус обработки:

```bash
curl -H "Authorization: Bearer $ROZEETA_API_KEY" \
  http://127.0.0.1:5010/api/v1/batches/<batch-id>
```

Получить результат:

```bash
curl -H "Authorization: Bearer $ROZEETA_API_KEY" \
  http://127.0.0.1:5010/api/v1/batches/<batch-id>/results
```

## Формат результата

У каждой записи есть технический статус обработки `status` и отдельный бизнес-статус анализа `analysis_status`.

Пример:

```json
{
  "filename": "call.mp3",
  "status": "done",
  "stage": "Готово",
  "progress": 100,
  "analysis_status": {
    "value": "target",
    "label": "целевой",
    "reason": "оснований для нецелевого не выявлено",
    "raw": "целевой — оснований для нецелевого не выявлено"
  },
  "segments": [],
  "transcript": "...",
  "llm": "...",
  "checklist": {},
  "soft_labels": "<audio_analysis>...</audio_analysis>",
  "speech_ranges_count": 42
}
```

Возможные значения `analysis_status.value`:

- `target` — целевой звонок;
- `not_target` — нецелевой звонок;
- `interested` — клиент заинтересован;
- `uncertain` — статус не определён;
- `unknown` — статус не удалось извлечь из ответа модели.

## Важные замечания

- Результаты batch-задач сейчас живут в памяти процесса. После перезапуска сервиса старые `batch_id` не восстановятся.
- Это сделано намеренно: предполагается, что внешний потребитель забирает `/results` и сохраняет данные в своём хранилище.
- Мягкие метки спикеров строятся после STT. Сервис берёт таймкоды слов, склеивает близкие слова в речевые диапазоны и запускает spectral-анализ только по этим диапазонам, чтобы меньше учитывать шум, музыку, тишину и хвосты записи.
- В Dockerfile используется один Gunicorn worker и несколько threads. Не увеличивайте количество worker-процессов, пока состояние batch-задач хранится в памяти.
- API-ключи хранятся в `data/api_keys.sqlite3`; в Docker Compose каталог `/app/data` вынесен в volume.
- Загруженные аудиофайлы временно лежат в `tmp_audio/` и удаляются после обработки.
- Настоящий `.env`, SQLite-база ключей, временные аудио и кэши исключены из git.

## Локальная разработка

Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Linux/macOS:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Проверка

```bash
python -m py_compile app.py api_keys.py config.py blueprints/*.py services/*.py
```

Если Docker daemon запущен:

```bash
docker build -t rozeeta:local .
```
