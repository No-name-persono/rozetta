# Rozeeta

Rozeeta is a small call-processing service for QA workflows. It accepts audio recordings, sends them through SpeechKit STT, enriches the transcript with optional soft speaker labels, asks an LLM to evaluate the call, and returns structured JSON with:

- processing status;
- business analysis status;
- transcript segments with timestamps;
- LLM summary;
- checklist answers with quotes and timestamps;
- optional soft speaker-label hints.

The app has a simple web UI, but the useful deployment shape is API-first: another system can upload audio, poll the batch, store the result wherever it already keeps call data, and ignore Rozeeta after processing.

## Quick Start With Docker

Create your environment file:

```bash
cp .env.example .env
```

Fill in Yandex Cloud, Object Storage, and LLM credentials in `.env`, then run:

```bash
docker compose up --build -d
```

Health check:

```bash
curl http://127.0.0.1:5010/api/v1/health
```

## API Keys

API keys are created from the terminal and are never shown in the web interface. Only SHA-256 hashes are stored in SQLite.

Inside Docker:

```bash
docker compose exec rozeeta python api_keys.py create my-integration
docker compose exec rozeeta python api_keys.py list
docker compose exec rozeeta python api_keys.py revoke 1
```

Locally:

```bash
python api_keys.py create my-integration
```

Save the generated key immediately. It is shown once.

## API

Use:

```http
Authorization: Bearer <api-key>
```

List prompts and checklists:

```bash
curl -H "Authorization: Bearer $ROZEETA_API_KEY" \
  http://127.0.0.1:5010/api/v1/templates
```

Create a batch:

```bash
curl -X POST http://127.0.0.1:5010/api/v1/batches \
  -H "Authorization: Bearer $ROZEETA_API_KEY" \
  -F "prompt_id=<prompt-id>" \
  -F "checklist_id=<checklist-id>" \
  -F "audio_files=@/path/to/call.mp3"
```

Poll processing status:

```bash
curl -H "Authorization: Bearer $ROZEETA_API_KEY" \
  http://127.0.0.1:5010/api/v1/batches/<batch-id>
```

Fetch results:

```bash
curl -H "Authorization: Bearer $ROZEETA_API_KEY" \
  http://127.0.0.1:5010/api/v1/batches/<batch-id>/results
```

## Result Shape

Each item has a technical processing status and a separate business analysis status:

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
  }
}
```

Known `analysis_status.value` values:

- `target`
- `not_target`
- `interested`
- `uncertain`
- `unknown`

## Runtime Notes

- Keep Gunicorn at one worker while batch state is in memory. The Dockerfile already does this.
- API keys are persisted in `data/api_keys.sqlite3`; Docker Compose stores `/app/data` in a named volume.
- Batch results are intentionally transient and live in process memory. Store completed API responses in your CRM, BI, data lake, or queue consumer.
- Uploaded audio is temporarily stored in `tmp_audio/` and deleted after processing.

## Development

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python app.py
```

On Linux/macOS:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```
