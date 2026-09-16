#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import boto3

ARTIFACT_FILES = ("model.onnx", "vectorizer.joblib", "metadata.json", "reference_stats.json")


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "https://minio.v-efimov.tech"),
    )


def download(bucket: str, prefix: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        client().download_file(bucket, f"{prefix.rstrip('/')}/{name}", str(destination / name))
        print(f"downloaded {name}")


def upload(bucket: str, prefix: str, source: Path) -> None:
    for name in ARTIFACT_FILES:
        client().upload_file(str(source / name), bucket, f"{prefix.rstrip('/')}/{name}")
        print(f"uploaded {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer model artifacts to/from S3")
    parser.add_argument("action", choices=("download", "upload"))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "m.babaev"))
    parser.add_argument("--prefix", default=os.getenv("S3_PREFIX", "project/artifacts/v1"))
    parser.add_argument("--directory", type=Path, default=Path("/artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.action == "download":
        download(arguments.bucket, arguments.prefix, arguments.directory)
    else:
        upload(arguments.bucket, arguments.prefix, arguments.directory)
