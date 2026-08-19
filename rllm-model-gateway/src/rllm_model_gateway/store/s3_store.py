"""S3-backed trace store.

Object layout (under an optional key prefix)::

    traces/{trace_id}.json
        Full serialized trace payload.

    sessions/{session_id}/{created_at:015.4f}-{trace_id}.json
        Lightweight marker used for session-scoped listing. Sorting the
        listing lexicographically yields traces in insertion order.

    consumed/sessions/{session_id}/...
        Where a session's markers + reward are moved once a reader has
        consumed it (see ``mark_consumed``). ``list_sessions`` scans only the
        live ``sessions/`` prefix, so its cost stays O(unconsumed) instead of
        O(all-sessions-ever) — the key to keeping polling scalable.

Reads issue ``ListObjectsV2`` on the session prefix, then fetch the
corresponding ``traces/{trace_id}.json`` objects. All boto3 calls are
executed in a thread executor so the interface stays async-compatible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3TraceStore:
    """Persistent trace store backed by an S3 bucket."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region_name: str | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3TraceStore requires a bucket name")
        self.bucket = bucket
        prefix = (prefix or "").strip("/")
        self.prefix = f"{prefix}/" if prefix else ""
        self._client = boto3.client(
            "s3",
            region_name=region_name or os.environ.get("AWS_REGION"),
        )

    def _trace_key(self, trace_id: str) -> str:
        return f"{self.prefix}traces/{trace_id}.json"

    def _reward_key(self, session_id: str) -> str:
        # Reward is one-per-session and small; store the full payload
        # directly under the session prefix (no separate index object).
        return f"{self.prefix}sessions/{session_id}/reward.json"

    def _session_marker_key(self, session_id: str, trace_id: str, created_at: float) -> str:
        return (
            f"{self.prefix}sessions/{session_id}/"
            f"{created_at:015.4f}-{trace_id}.json"
        )

    def _session_prefix(self, session_id: str) -> str:
        return f"{self.prefix}sessions/{session_id}/"

    def _consumed_session_prefix(self, session_id: str) -> str:
        return f"{self.prefix}consumed/sessions/{session_id}/"

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        now = time.time()
        payload = json.dumps(data).encode("utf-8")
        if trace_id.startswith("reward-"):
            # Single-object write: full reward payload at sessions/{sid}/reward.json.
            await asyncio.to_thread(self._put_reward, session_id, now, payload)
            return
        marker = json.dumps({"trace_id": trace_id, "created_at": now}).encode("utf-8")
        await asyncio.to_thread(
            self._put_pair, trace_id, session_id, now, payload, marker
        )

    def _put_reward(self, session_id: str, created_at: float, payload: bytes) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._reward_key(session_id),
            Body=payload,
            ContentType="application/json",
            Metadata={"created_at": f"{created_at}"},
        )

    def _put_pair(
        self,
        trace_id: str,
        session_id: str,
        created_at: float,
        payload: bytes,
        marker: bytes,
    ) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._trace_key(trace_id),
            Body=payload,
            ContentType="application/json",
            Metadata={"created_at": f"{created_at}"},
        )
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._session_marker_key(session_id, trace_id, created_at),
            Body=marker,
            ContentType="application/json",
            Metadata={"trace_id": trace_id, "created_at": f"{created_at}"},
        )

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_trace_sync, trace_id)

    def _get_object_json(self, key: str) -> dict[str, Any] | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(resp["Body"].read())

    def _get_trace_sync(self, trace_id: str) -> dict[str, Any] | None:
        # ``trace_id`` alone doesn't identify a session, so a bare
        # ``GET /traces/{trace_id}`` call for a reward id can't be
        # resolved here — reward reads happen via ``get_session_traces``.
        return self._get_object_json(self._trace_key(trace_id))

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        markers = await asyncio.to_thread(self._list_markers, session_id, since, limit)
        fetches = []
        for m in markers:
            if m["trace_id"].startswith("reward-"):
                # Reward payload lives at sessions/{sid}/reward.json, not traces/.
                fetches.append(
                    asyncio.to_thread(self._get_object_json, self._reward_key(session_id))
                )
            else:
                fetches.append(asyncio.to_thread(self._get_trace_sync, m["trace_id"]))
        results = await asyncio.gather(*fetches)
        return [t for t in results if t is not None]

    def _list_markers(
        self,
        session_id: str,
        since: float | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        prefix = self._session_prefix(session_id)
        paginator = self._client.get_paginator("list_objects_v2")
        markers: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                filename = obj["Key"][len(prefix):]
                if not filename.endswith(".json"):
                    continue
                base = filename[: -len(".json")]
                # Fixed-key reward marker: sessions/{sid}/reward.json.
                # LastModified is the S3 object's write time; use it as
                # created_at so listings still sort chronologically.
                if base == "reward":
                    trace_id = f"reward-{session_id}"
                    created_at = obj["LastModified"].timestamp()
                else:
                    ts_str, _, trace_id = base.partition("-")
                    try:
                        created_at = float(ts_str)
                    except ValueError:
                        continue
                if since is not None and created_at < since:
                    continue
                markers.append({"trace_id": trace_id, "created_at": created_at})
                if limit is not None and len(markers) >= limit:
                    return markers
        return markers

    async def delete_session(self, session_id: str) -> int:
        return await asyncio.to_thread(self._delete_session_sync, session_id)

    def _delete_session_sync(self, session_id: str) -> int:
        prefix = self._session_prefix(session_id)
        paginator = self._client.get_paginator("list_objects_v2")
        trace_ids: list[str] = []
        session_keys: list[str] = []
        reward_present = False
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                session_keys.append(obj["Key"])
                filename = obj["Key"][len(prefix):]
                if not filename.endswith(".json"):
                    continue
                base = filename[: -len(".json")]
                if base == "reward":
                    reward_present = True
                else:
                    _, _, trace_id = base.partition("-")
                    if trace_id:
                        trace_ids.append(trace_id)
        # Session prefix holds markers + (optionally) the reward payload.
        self._batch_delete(session_keys)
        # Trace payloads live under traces/{tid}.json.
        self._batch_delete([self._trace_key(tid) for tid in trace_ids])
        return len(trace_ids) + (1 if reward_present else 0)

    def _batch_delete(self, keys: list[str]) -> None:
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            if not chunk:
                continue
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )

    async def mark_consumed(self, session_id: str) -> int:
        """Move a session's marker + reward objects out of the live listing.

        Relocates every object under ``sessions/{sid}/`` to
        ``consumed/sessions/{sid}/`` (copy + delete). Trace payloads under
        ``traces/`` are left in place — ``list_sessions`` never scans them,
        so moving only the small marker/reward objects is enough to shrink
        the scan while preserving full history for later inspection.

        Idempotent: a session already moved (no live objects) is a no-op.
        Returns the number of objects relocated.
        """
        return await asyncio.to_thread(self._mark_consumed_sync, session_id)

    def _mark_consumed_sync(self, session_id: str) -> int:
        src_prefix = self._session_prefix(session_id)
        dst_prefix = self._consumed_session_prefix(session_id)
        paginator = self._client.get_paginator("list_objects_v2")
        moved_keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=src_prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                dst_key = dst_prefix + key[len(src_prefix):]
                self._client.copy_object(
                    Bucket=self.bucket,
                    CopySource={"Bucket": self.bucket, "Key": key},
                    Key=dst_key,
                )
                moved_keys.append(key)
        # Delete originals only after all copies succeeded (best-effort atomicity).
        self._batch_delete(moved_keys)
        return len(moved_keys)

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_sessions_sync, since, limit)

    def _list_sessions_sync(
        self,
        since: float | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        base = f"{self.prefix}sessions/"
        paginator = self._client.get_paginator("list_objects_v2")
        sessions: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=base, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []) or []:
                sid = cp.get("Prefix", "")[len(base):].rstrip("/")
                if not sid:
                    continue
                markers = self._list_markers(sid, since=None, limit=None)
                if not markers:
                    continue
                first_at = min(m["created_at"] for m in markers)
                if since is not None and first_at < since:
                    continue
                sessions.append(
                    {
                        "session_id": sid,
                        "trace_count": len(markers),
                        "first_trace_at": first_at,
                        "last_trace_at": max(m["created_at"] for m in markers),
                    }
                )
        sessions.sort(key=lambda r: r["first_trace_at"], reverse=True)
        if limit is not None:
            sessions = sessions[:limit]
        return sessions

    async def flush(self) -> None:
        """No-op: S3 puts are durable on return."""

    async def close(self) -> None:
        """No-op: boto3 clients don't hold long-lived connections we need to release."""
