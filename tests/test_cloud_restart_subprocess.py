"""Durable-queue survives a *real* process crash (Part 3).

The in-process ``del q`` test already proves the recovery logic; this stricter test
proves durability does not depend on ``atexit``/GC/clean shutdown by spawning a
child process that writes batches and then calls ``os._exit(0)`` — which bypasses
all finalizers, simulating a hard crash / power loss.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")

# Child: write N batches durably, then hard-exit without any cleanup.
_CHILD = """
import os, sys
sys.path.insert(0, sys.argv[1])
from fibersail_edge.cloud import DurableQueue
q = DurableQueue(sys.argv[2])
for i in range(int(sys.argv[3])):
    q.put(("batch-%d" % i).encode())
sys.stdout.flush()
os._exit(0)   # hard crash: no finalizers, no clean shutdown
"""


def test_hard_crash_survives_and_recovers_in_fifo_order() -> None:
    with tempfile.TemporaryDirectory() as spool:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD, _SRC, spool, "6"],
            capture_output=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr.decode()

        # Parent recovers on the same spool after the child's hard exit.
        from fibersail_edge.cloud import DurableQueue

        q = DurableQueue(spool)
        assert len(q) == 6
        drained = []
        while (item := q.peek()) is not None:
            drained.append(item[1])
            q.ack(item[0])
        assert drained == [f"batch-{i}".encode() for i in range(6)]
