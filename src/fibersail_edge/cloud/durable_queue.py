"""A crash-safe, restart-recoverable file-backed FIFO queue (stdlib only).

This is the durability boundary of Part 3: a batch handed to :meth:`put` is on
stable storage before the call returns, so it survives a process crash / restart.
The design is deliberately small and boring, because durability bugs hide in
cleverness.

Crash-safety argument
----------------------
* **Atomic publish.** A batch is written to ``<seq>.batch.tmp``, ``fsync``'d, then
  ``os.replace``'d to ``<seq>.batch``. ``os.replace`` is atomic on POSIX and
  Windows, so a concurrent reader (or a crash) ever sees either no file or the
  complete file — never a half-written one.
* **Data before name.** ``fsync(file)`` before the rename guarantees the bytes are
  on the platter before the name becomes visible (otherwise a crash could leave a
  correctly-named file full of zeros). ``fsync(dir)`` after the rename makes the
  rename itself survive power loss.
* **Recovery.** On construction we scan the spool: orphan ``*.tmp`` files (writes
  that never atomically published) are deleted, committed ``*.batch`` files are the
  queue, and the next sequence continues from ``max(seq)+1`` — no separate counter
  file to disagree with the spool after a crash.

Delivery is **at-least-once**: :meth:`peek` returns the oldest batch without
removing it; the consumer uploads, then :meth:`ack`\\ s (deletes) it. A crash
between a successful upload and its ack re-delivers that batch on restart — made
harmless upstream by deterministic, idempotent S3 keys.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

_SUFFIX = ".batch"
_TMP_SUFFIX = ".batch.tmp"
_SEQ_WIDTH = 20  # zero-padded so lexical order == numeric order == FIFO; covers 64-bit


class DurableQueue:
    """A durable FIFO of opaque ``bytes`` batches, one file per batch.

    Args:
        spool_dir: Directory holding the spool (created if missing).
        fsync: ``fsync`` each batch file before publishing (real durability). Tests
            disable it for speed.
        fsync_dir: ``fsync`` the directory after each rename/remove (makes the
            rename durable). Silently skipped where the platform rejects it.
        max_bytes / max_files: Optional disk cap. When exceeded, the oldest batches
            are dropped (and counted in :attr:`dropped_batches`) so a long outage
            can't fill the disk. ``None`` (default) = uncapped.
    """

    def __init__(
        self,
        spool_dir: str,
        *,
        fsync: bool = True,
        fsync_dir: bool = True,
        max_bytes: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> None:
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be > 0 when set")
        if max_files is not None and max_files <= 0:
            raise ValueError("max_files must be > 0 when set")
        self._dir = spool_dir
        self._fsync = fsync
        self._fsync_dir = fsync_dir
        self._max_bytes = max_bytes
        self._max_files = max_files
        self._lock = threading.Lock()
        self._next_seq = 0
        self._count = 0
        self._total_bytes = 0
        self._dropped = 0
        self._recover()

    # -- Introspection ---------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return self._count

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def dropped_batches(self) -> int:
        """Number of batches evicted by the disk cap over this queue's lifetime."""
        with self._lock:
            return self._dropped

    # -- Queue operations ------------------------------------------------------

    def put(self, data: bytes) -> str:
        """Durably append a batch. Returns its id; the batch is on disk on return."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            self._enforce_cap_locked(len(data))
        name = f"{seq:0{_SEQ_WIDTH}d}{_SUFFIX}"
        final = os.path.join(self._dir, name)
        tmp = os.path.join(self._dir, f"{seq:0{_SEQ_WIDTH}d}{_TMP_SUFFIX}")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            if self._fsync:
                os.fsync(fh.fileno())
        os.replace(tmp, final)  # atomic publish
        if self._fsync_dir:
            self._fsync_directory()
        with self._lock:
            self._count += 1
            self._total_bytes += len(data)
        return name

    def peek(self) -> Optional[Tuple[str, bytes]]:
        """Return ``(id, data)`` for the oldest committed batch, or ``None`` if empty."""
        with self._lock:
            files = self._committed_files()
        if not files:
            return None
        name = files[0]
        try:
            with open(os.path.join(self._dir, name), "rb") as fh:
                return (name, fh.read())
        except FileNotFoundError:
            return None  # raced with a concurrent ack; caller simply re-peeks

    def ack(self, batch_id: str) -> None:
        """Remove an uploaded batch. Idempotent (a missing file is a no-op)."""
        path = os.path.join(self._dir, batch_id)
        try:
            size = os.path.getsize(path)
            os.remove(path)
        except FileNotFoundError:
            return
        with self._lock:
            self._count -= 1
            self._total_bytes -= size

    # -- Internals -------------------------------------------------------------

    def _committed_files(self) -> List[str]:
        """Committed batch filenames, oldest-first (lexical == numeric via padding)."""
        return sorted(n for n in os.listdir(self._dir) if n.endswith(_SUFFIX))

    def _recover(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        seqs: List[int] = []
        total = 0
        for name in os.listdir(self._dir):
            if name.endswith(_TMP_SUFFIX):
                # Orphaned in-progress write that never atomically published — drop it.
                _remove_quietly(os.path.join(self._dir, name))
            elif name.endswith(_SUFFIX):
                try:
                    seq = int(name[: -len(_SUFFIX)])
                    size = os.path.getsize(os.path.join(self._dir, name))
                except ValueError:
                    continue  # ignore foreign files
                except FileNotFoundError:
                    continue  # deleted concurrently (e.g. a consumer acked it) mid-scan
                seqs.append(seq)
                total += size
        self._count = len(seqs)
        self._total_bytes = total
        self._next_seq = max(seqs) + 1 if seqs else 0

    def _enforce_cap_locked(self, incoming_size: int) -> None:
        if self._max_files is not None:
            while self._count + 1 > self._max_files and self._count > 0:
                self._drop_oldest_locked()
        if self._max_bytes is not None:
            while self._total_bytes + incoming_size > self._max_bytes and self._count > 0:
                self._drop_oldest_locked()

    def _drop_oldest_locked(self) -> None:
        files = self._committed_files()
        if not files:
            return
        path = os.path.join(self._dir, files[0])
        try:
            size = os.path.getsize(path)
            os.remove(path)
        except FileNotFoundError:
            return
        self._count -= 1
        self._total_bytes -= size
        self._dropped += 1

    def _fsync_directory(self) -> None:
        try:
            fd = os.open(self._dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # directory fsync is unsupported on some platforms/filesystems


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


__all__ = ["DurableQueue"]
