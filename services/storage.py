
import uuid
import os
from datetime import datetime
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from config import DEFAULTS

def s3_client(addressing_style: str):
    return boto3.client(
        "s3",
        endpoint_url=DEFAULTS["S3_ENDPOINT"],
        aws_access_key_id=(DEFAULTS["S3_ACCESS_KEY"] or "").strip(),
        aws_secret_access_key=(DEFAULTS["S3_SECRET_KEY"] or "").strip(),
        region_name=DEFAULTS["S3_REGION"],
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": addressing_style}),
    )

def upload_local_file(local_path: str, ext: str) -> tuple[str, str, str]:
    """
    Совместимая с вашим скриптом загрузка в Object Storage.
    Возвращает (uri, s3_key, addressing_style).
    """
    key = f"speech_web/{datetime.utcnow().strftime('%Y%m%d')}/{uuid.uuid4().hex}{ext}"
    last_err = None
    # Порядок как в вашем рабочем скрипте: сначала virtual, затем path
    for style in ("virtual", "path"):
        try:
            s3_client(style).upload_file(local_path, DEFAULTS["S3_BUCKET"], key)
            uri = f"{DEFAULTS['S3_ENDPOINT'].rstrip('/')}/{DEFAULTS['S3_BUCKET']}/{key}"
            return uri, key, style
        except ClientError as e:
            last_err = e
            continue
    raise last_err  # type: ignore

def delete_remote(key: str, style: str):
    try:
        s3_client(style).delete_object(Bucket=DEFAULTS["S3_BUCKET"], Key=key)
    except Exception:
        pass
