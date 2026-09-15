"""Thin MinIO (S3-compatible) client wrapper shared by every pipeline stage."""
from __future__ import annotations

from pathlib import Path

from minio import Minio

from src.common.config import get_settings


def get_minio_client() -> Minio:
    settings = get_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def upload_file(local_path: Path, bucket: str, object_name: str) -> str:
    client = get_minio_client()
    client.fput_object(bucket, object_name, str(local_path))
    return f"s3://{bucket}/{object_name}"


def download_file(bucket: str, object_name: str, local_path: Path) -> Path:
    client = get_minio_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.fget_object(bucket, object_name, str(local_path))
    return local_path
