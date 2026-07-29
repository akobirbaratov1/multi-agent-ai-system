"""
Durable JSON / JSONL storage helpers.

The file-backed stores in this project are written from concurrent request
handlers. Plain `open(path, "w")` truncates the file before the new content is
written, so an interrupted or interleaved write leaves a corrupted document
behind. Every write here goes to a temporary file in the same directory and is
then atomically renamed over the target, guarded by a per-path lock.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List

_locks: Dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def lock_for(path: Path):
    """
    Return the process-wide lock guarding a given file path.

    Re-entrant on purpose: callers routinely wrap a read-modify-write in
    `lock_for(path)` and then call `write_json`, which takes the same lock
    again. A plain Lock deadlocks the thread against itself there.
    """
    key = str(path)
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.RLock()
        return _locks[key]


def read_json(path: Path, default: Any) -> Any:
    """
    Read a JSON document, returning `default` when the file is missing.

    A corrupted file (truncated by an older non-atomic writer, or hand-edited)
    also yields the default rather than taking the caller down.
    """
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Atomically write a JSON document, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with lock_for(path):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write raw bytes (used for the FAISS index file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with lock_for(path):
        try:
            with open(tmp, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record to a JSONL file under the path lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with lock_for(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    """
    Stream records from a JSONL file, skipping malformed lines.

    Streaming rather than returning a list keeps memory flat as the log grows;
    callers that genuinely need every record can still materialise it.
    """
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file fully into a list."""
    return list(iter_jsonl(path))


def tail_jsonl(path: Path, limit: int) -> List[dict]:
    """Return at most the last `limit` records without holding the whole file."""
    from collections import deque

    return list(deque(iter_jsonl(path), maxlen=limit))
