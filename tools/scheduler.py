"""
Background scheduler for due follow-ups.

`schedule_followup` wrote records to a queue that nothing ever drained —
`process_due_followups` existed but had no caller, so scheduled follow-ups
never went out. This runs the sweep on an interval alongside the API.

Set FOLLOWUP_INTERVAL_SECONDS to 0 to disable (the default in demo mode and
tests). In a multi-replica deployment, run it in exactly one replica — or
move the queue to a store with real leasing.
"""

import asyncio
import os
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)


def _interval() -> int:
    try:
        return int(os.getenv("FOLLOWUP_INTERVAL_SECONDS", "0"))
    except ValueError:
        return 0


class FollowUpScheduler:
    def __init__(self, interval_seconds: Optional[int] = None):
        self.interval = _interval() if interval_seconds is None else interval_seconds
        self._task: Optional[asyncio.Task] = None

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    async def _run(self) -> None:
        from tools.followup import process_due_followups

        while True:
            try:
                await asyncio.sleep(self.interval)
                # The sweep does blocking file IO and mock sends; keep it off
                # the event loop so request handling isn't stalled.
                summary = await asyncio.to_thread(process_due_followups)
                if any(summary.values()):
                    logger.info("Processed due follow-ups", extra=summary)
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad sweep must not kill the loop for the process lifetime.
                logger.exception("Follow-up sweep failed; continuing")

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info("Follow-up scheduler started", extra={"interval_seconds": self.interval})

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
