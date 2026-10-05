"""S3 helpers shared by every tool.

Artifacts move by URI rather than by value, so a tool downloads its inputs into
scratch, runs, and uploads what it produced. Each call writes into one flat
folder of its own:

    s3://$FRIDAY_S3_BUCKET/runs/<run_id>/designs.fa

We upload explicitly with boto3 instead of mounting the bucket. Modal's
CloudBucketMount forbids append mode and forbids seek combined with write, which
breaks `numpy.savez`, and the files here are written by ProteinMPNN itself
rather than by us, so its write patterns are not ours to constrain.

`parse_uri`, `run_prefix`, `join` and `new_run_id` are pure and need no
credentials. Only `download` and `upload` touch AWS.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

BUCKET_ENV = "FRIDAY_S3_BUCKET"
PREFIX_ENV = "FRIDAY_S3_PREFIX"
DEFAULT_PREFIX = "runs"

_URI = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def join(self, *parts: str) -> "S3Uri":
        """Append path segments, e.g. a run prefix plus a file name."""
        tail = "/".join(p.strip("/") for p in parts if p.strip("/"))
        return S3Uri(self.bucket, f"{self.key.rstrip('/')}/{tail}" if tail else self.key)


def parse_uri(uri: str) -> S3Uri:
    """Split an `s3://bucket/key` URI, rejecting anything else."""
    match = _URI.match(uri.strip())
    if not match:
        raise ValueError(f"Not an S3 URI: {uri!r}. Expected s3://bucket/key")
    return S3Uri(match["bucket"], match["key"])


def new_run_id() -> str:
    """A time-ordered id, e.g. `20260909T142530-a3f9c1b2`.

    Sorts chronologically as an S3 prefix, so a bucket listing reads as a
    history, and stays legible in logs and error messages.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def bucket() -> str:
    name = os.environ.get(BUCKET_ENV, "").strip()
    if not name:
        raise RuntimeError(
            f"{BUCKET_ENV} is not set. It comes from the `friday-aws` Modal "
            f"secret, so add it there, or export it to run locally."
        )
    return name


def run_prefix(run_id: str) -> S3Uri:
    """The one folder a single call writes its artifacts into."""
    root = os.environ.get(PREFIX_ENV, DEFAULT_PREFIX).strip("/") or DEFAULT_PREFIX
    return S3Uri(bucket(), f"{root}/{run_id}")


@cache
def client():
    """A cached boto3 S3 client. Credentials come from the environment."""
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION") or None)


def download(uri: str, dest: Path) -> Path:
    """Fetch an object to a local path, creating parent directories."""
    parsed = parse_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"get {uri} -> {dest}", flush=True)
    client().download_file(parsed.bucket, parsed.key, str(dest))
    return dest


def exists(uri: str | S3Uri) -> bool:
    """Whether an object is there, without downloading it."""
    from botocore.exceptions import ClientError

    parsed = parse_uri(str(uri))
    try:
        client().head_object(Bucket=parsed.bucket, Key=parsed.key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise


def upload(path: Path, uri: str | S3Uri) -> str:
    """Store a local file, returning the URI it was written to."""
    target = parse_uri(str(uri))
    print(f"put {path} -> {target}", flush=True)
    client().upload_file(str(path), target.bucket, target.key)
    return str(target)
