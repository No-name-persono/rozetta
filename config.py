import os
try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

DEFAULTS = {
    # S3
    "S3_ENDPOINT": os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net"),
    "S3_REGION": os.getenv("S3_REGION", "ru-central1"),
    "S3_BUCKET": os.getenv("S3_BUCKET", ""),
    "S3_ACCESS_KEY": os.getenv("S3_ACCESS_KEY", ""),
    "S3_SECRET_KEY": os.getenv("S3_SECRET_KEY", ""),

    # SpeechKit
    "YANDEX_API_KEY": os.getenv("YANDEX_API_KEY", ""),
    "YANDEX_IAM_TOKEN": os.getenv("YANDEX_IAM_TOKEN", ""),
    "YANDEX_FOLDER_ID": os.getenv("YANDEX_FOLDER_ID", ""),
    "ASR_API_VERSION": os.getenv("ASR_API_VERSION", "v3").lower(),
    "ASR_MODEL": os.getenv("ASR_MODEL", "general"),
    "ASR_LANGUAGE": os.getenv("ASR_LANGUAGE", "ru-RU"),
    "ASR_LITERATURE_TEXT": os.getenv("ASR_LITERATURE_TEXT", "1") in {"1", "true", "True"},
    "ASR_ENDPOINT": os.getenv("ASR_ENDPOINT", "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"),
    "ASR_V3_ENDPOINT": os.getenv("ASR_V3_ENDPOINT", "https://stt.api.cloud.yandex.net/stt/v3/recognizeFileAsync"),
    "ASR_V3_RESULT_ENDPOINT": os.getenv("ASR_V3_RESULT_ENDPOINT", "https://stt.api.cloud.yandex.net:443/stt/v3/getRecognition"),
    "OPS_ENDPOINT": os.getenv("OPS_ENDPOINT", "https://operation.api.cloud.yandex.net/operations/{}"),

    # LLM (YandexGPT)
    "LLM_API_URL": os.getenv("LLM_API_URL", "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"),
    "LLM_MODEL_URI": os.getenv("LLM_MODEL_URI", ""),
    "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
    "LLM_IAM_TOKEN": os.getenv("LLM_IAM_TOKEN", ""),
    "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS", "3500"),

    # Spectral diarization
    "SPECTRAL_ENABLED": os.getenv("SPECTRAL_ENABLED", "1") in {"1", "true", "True"},
    "SPECTRAL_N_SPEAKERS": int(os.getenv("SPECTRAL_N_SPEAKERS", "2")),
    "SPECTRAL_IVR_CUTOFF_SEC": float(os.getenv("SPECTRAL_IVR_CUTOFF_SEC", "0")),
    "SPECTRAL_CONFIDENCE_CUTOFF": float(os.getenv("SPECTRAL_CONFIDENCE_CUTOFF", "0.55")),
    "SPECTRAL_METHOD": os.getenv("SPECTRAL_METHOD", "windowed"),
    "SPECTRAL_GROUP_SEC": float(os.getenv("SPECTRAL_GROUP_SEC", "60")),
    "SPECTRAL_WINDOW_SEC": float(os.getenv("SPECTRAL_WINDOW_SEC", "3.5")),
    "SPECTRAL_STEP_SEC": float(os.getenv("SPECTRAL_STEP_SEC", "0.5")),
    "SPECTRAL_MIN_BLOCK_SEC": float(os.getenv("SPECTRAL_MIN_BLOCK_SEC", "0")),
    "SPECTRAL_SHORT_UNCERTAIN_SEC": float(os.getenv("SPECTRAL_SHORT_UNCERTAIN_SEC", "2.5")),
    "SPECTRAL_ANCHOR_MIN_SEC": float(os.getenv("SPECTRAL_ANCHOR_MIN_SEC", "8")),
    "SPECTRAL_MIXED_CHECK_SEC": float(os.getenv("SPECTRAL_MIXED_CHECK_SEC", "0")),
    "SPECTRAL_MIXED_MIN_PART_SEC": float(os.getenv("SPECTRAL_MIXED_MIN_PART_SEC", "1.2")),
    "SPECTRAL_MICRO_WINDOW_SEC": float(os.getenv("SPECTRAL_MICRO_WINDOW_SEC", "1.2")),
    "SPECTRAL_MICRO_STEP_SEC": float(os.getenv("SPECTRAL_MICRO_STEP_SEC", "0.25")),
    "SPECTRAL_TRANSIENT_ENABLED": os.getenv("SPECTRAL_TRANSIENT_ENABLED", "1") in {"1", "true", "True"},
    "SPECTRAL_TRANSIENT_SEARCH_SEC": float(os.getenv("SPECTRAL_TRANSIENT_SEARCH_SEC", "60")),
    "SPECTRAL_TRANSIENT_LATE_START_SEC": float(os.getenv("SPECTRAL_TRANSIENT_LATE_START_SEC", "75")),
    "SPECTRAL_TRANSIENT_MIN_SEC": float(os.getenv("SPECTRAL_TRANSIENT_MIN_SEC", "4")),
    "SPECTRAL_TRANSIENT_DISTANCE_THRESHOLD": float(os.getenv("SPECTRAL_TRANSIENT_DISTANCE_THRESHOLD", "9")),
    "SPECTRAL_TRANSIENT_LOCAL_SPEAKERS": int(os.getenv("SPECTRAL_TRANSIENT_LOCAL_SPEAKERS", "3")),

    # App
    "SECRET_KEY": os.getenv("SECRET_KEY", "dev"),
    "DELETE_REMOTE_AFTER": os.getenv("DELETE_REMOTE_AFTER", "1") in {"1","true","True"},
    "JOB_TTL_SEC": int(os.getenv("JOB_TTL_SEC", "7200")),
    "MAX_UPLOAD_MB": int(os.getenv("MAX_UPLOAD_MB", "64")),
    "ASYNC_MAX_WORKERS": int(os.getenv("ASYNC_MAX_WORKERS", "4")),
    "ASR_POLL_INTERVAL_SEC": float(os.getenv("ASR_POLL_INTERVAL_SEC", "3")),
    "ASR_POLL_TIMEOUT_SEC": int(os.getenv("ASR_POLL_TIMEOUT_SEC", "7200")),
    "API_KEYS_DB": os.getenv("API_KEYS_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "api_keys.sqlite3")),
}

ALLOWED_EXTS = {".mp3", ".ogg", ".opus"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "tmp_audio")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATES_PATH = os.path.join(DATA_DIR, "qa_templates.json")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
