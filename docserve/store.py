"""Redis-backed result cache and job queue, plus document storage (local disk or S3)."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)

QUEUE_KEY = "docserve:jobs:queue"
JOB_KEY = "docserve:job:{}"
CACHE_KEY = "docserve:cache:{}"
BLOB_KEY = "docserve:blob:{}"


class MemoryCache:
    """Bounded LRU used when Redis is not configured."""

    def __init__(self, max_items: int = 512):
        self._d: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.max_items = max_items

    def get(self, key: str) -> dict | None:
        item = self._d.get(key)
        if not item or item[0] < time.time():
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return json.loads(item[1])

    def set(self, key: str, value: dict, ttl: int) -> None:
        self._d[key] = (time.time() + ttl, json.dumps(value))
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

    def ping(self) -> bool:
        return True


class RedisCache:
    def __init__(self, client):
        self.r = client

    def get(self, key: str) -> dict | None:
        raw = self.r.get(CACHE_KEY.format(key))
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict, ttl: int) -> None:
        self.r.set(CACHE_KEY.format(key), json.dumps(value), ex=ttl)

    def ping(self) -> bool:
        return bool(self.r.ping())


class JobQueue:
    """FIFO job queue in a Redis list; job state and payload live in TTL'd keys.

    The API LPUSHes job ids; workers BRPOP one and then drain up to N-1 more
    without blocking, so documents are processed in batches under load.
    """

    def __init__(self, client, ttl: int = 86400):
        self.r, self.ttl = client, ttl

    def enqueue(self, image: bytes, params: dict) -> str:
        job_id = uuid.uuid4().hex
        pipe = self.r.pipeline()
        pipe.set(BLOB_KEY.format(job_id), image, ex=self.ttl)
        pipe.set(JOB_KEY.format(job_id), json.dumps({"id": job_id, "status": "queued", "params": params,
                                                     "created_at": time.time()}), ex=self.ttl)
        pipe.lpush(QUEUE_KEY, job_id)
        pipe.execute()
        return job_id

    def get(self, job_id: str) -> dict | None:
        raw = self.r.get(JOB_KEY.format(job_id))
        return json.loads(raw) if raw else None

    def update(self, job_id: str, **fields) -> None:
        job = self.get(job_id) or {"id": job_id}
        job.update(fields)
        self.r.set(JOB_KEY.format(job_id), json.dumps(job), ex=self.ttl)

    def blob(self, job_id: str) -> bytes | None:
        return self.r.get(BLOB_KEY.format(job_id))

    def drop_blob(self, job_id: str) -> None:
        self.r.delete(BLOB_KEY.format(job_id))

    def depth(self) -> int:
        return int(self.r.llen(QUEUE_KEY))

    def pop_batch(self, max_items: int, timeout: int = 2) -> list[str]:
        first = self.r.brpop(QUEUE_KEY, timeout=timeout)
        if not first:
            return []
        ids = [_s(first[1])]
        while len(ids) < max_items:
            nxt = self.r.rpop(QUEUE_KEY)
            if nxt is None:
                break
            ids.append(_s(nxt))
        return ids


def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else v


def redis_client(url: str):
    import redis

    return redis.Redis.from_url(url, socket_timeout=5, socket_connect_timeout=3)


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, key: str, data: bytes) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)


class S3Storage:
    """Stores uploaded documents / results in S3 (server-side encrypted, since documents carry PII)."""

    def __init__(self, bucket: str, prefix: str):
        import boto3

        self.s3, self.bucket, self.prefix = boto3.client("s3"), bucket, prefix.strip("/")

    def put(self, key: str, data: bytes) -> str:
        full = f"{self.prefix}/{key}"
        self.s3.put_object(Bucket=self.bucket, Key=full, Body=data, ServerSideEncryption="AES256")
        return f"s3://{self.bucket}/{full}"


def make_storage(settings):
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("DOCSERVE_S3_BUCKET is required when DOCSERVE_STORAGE_BACKEND=s3")
        return S3Storage(settings.s3_bucket, settings.s3_prefix)
    return LocalStorage(settings.storage_dir)
