from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.session.call import CallSession

if TYPE_CHECKING:
    from app.runtime import VoiceRuntime

logger = logging.getLogger("mysmartfood-voice.sessions")


class SessionManager:
    def __init__(self, runtime: VoiceRuntime) -> None:
        self._runtime = runtime
        self._sessions: dict[int, CallSession] = {}

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def create(self, websocket) -> CallSession:
        session = CallSession(websocket, self._runtime)
        self._sessions[id(websocket)] = session
        logger.info("step=session.open active=%s", self.active_count)
        return session

    async def close(self, websocket, reason: str = "disconnect") -> None:
        session = self._sessions.pop(id(websocket), None)
        if session is None:
            logger.info("step=session.already_removed reason=%s active=%s", reason, self.active_count)
            return
        try:
            await session.close(reason=reason)
        except Exception:
            logger.exception("step=session.close.error stream=%s", session.stream_sid)
        logger.info("step=session.removed reason=%s active=%s", reason, self.active_count)

    async def close_all(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            try:
                await session.close(reason="shutdown")
            except Exception:
                logger.exception("step=session.close.error stream=%s", session.stream_sid)
        if sessions:
            logger.info("step=session.close_all count=%s", len(sessions))
