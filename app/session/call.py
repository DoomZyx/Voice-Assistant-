"""Session vocale isolee : buffer, VAD, tour de parole, barge-in minimal."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from app.audio.buffer import AudioBuffer
from app.audio.codec import mulaw_b64_to_pcm16, pcm16_to_mulaw, split_mulaw_frames
from app.audio.resample import resample_pcm16
from app.llm import LLMResult
from app.llm.tools import LLM_TOOLS
from app.prompts import AGENT_INSTRUCTIONS
from app.twilio.protocol import clear_stream, media_out, parse_event

if TYPE_CHECKING:
    from app.runtime import VoiceRuntime

logger = logging.getLogger("mysmartfood-voice.session")

FALLBACK_REPLY = "Je n'ai pas compris. Pouvez-vous repeter ?"
MIN_UTTERANCE_SAMPLES_8K = 1600  # 200 ms
MEDIA_LOG_EVERY = 50


class CallSession:
    def __init__(self, websocket, runtime: VoiceRuntime) -> None:
        self._ws = websocket
        self._runtime = runtime
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self._history: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_INSTRUCTIONS.strip()},
        ]
        self._pre_roll = AudioBuffer()
        self._utterance = AudioBuffer()
        self._send_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._cancel = asyncio.Event()
        self._turn_task: asyncio.Task | None = None
        self._assistant_speaking = False
        self._closed = False
        self._media_in = 0
        self._media_out = 0
        self._vad = runtime.vad_model.new_session(runtime.settings) if runtime.vad_model else None

    @property
    def closed(self) -> bool:
        return self._closed

    def _log(self, step: str, **fields: Any) -> None:
        parts = [f"step={step}", f"stream={self.stream_sid}", f"call={self.call_sid}"]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        logger.info(" ".join(parts))

    async def handle_raw(self, raw: str | bytes) -> bool:
        """Traite un message Twilio. True = stream termine, le WS doit se fermer."""
        if self._closed:
            return True
        try:
            data = parse_event(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            self._log("twilio.bad_message", error=str(exc)[:200])
            return False
        event = data.get("event")
        if event == "connected":
            self._log("twilio.connected", protocol=data.get("protocol"), version=data.get("version"))
            return False
        if event == "start":
            await self._on_start(data)
            return False
        if event == "media":
            await self._on_media(data)
            return False
        if event == "mark":
            mark = data.get("mark") or {}
            self._log("twilio.mark", name=mark.get("name"))
            return False
        if event in {"stop", "closed"}:
            self._log("twilio.stop", media_in=self._media_in, media_out=self._media_out)
            return True
        self._log("twilio.event", event=event)
        return False

    async def close(self, reason: str = "disconnect") -> None:
        if self._closed:
            self._log("session.close.idempotent", reason=reason)
            return
        self._closed = True
        self._cancel.set()
        task = self._turn_task
        self._turn_task = None
        task_cancelled = False
        if task is not None and not task.done():
            task_cancelled = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("step=session.task_error stream=%s", self.stream_sid)
        buffered = len(self._utterance) + len(self._pre_roll)
        self._utterance.clear()
        self._pre_roll.clear()
        history_len = len(self._history)
        self._history.clear()
        if self._vad is not None:
            reset = getattr(self._vad, "reset", None)
            if callable(reset):
                reset()
        self._assistant_speaking = False
        self._ws = None
        self._log(
            "session.close",
            reason=reason,
            task_cancelled=task_cancelled,
            buffered_samples=buffered,
            history=history_len,
            media_in=self._media_in,
            media_out=self._media_out,
        )

    async def _on_start(self, data: dict[str, Any]) -> None:
        start = data.get("start") or {}
        self.stream_sid = start.get("streamSid")
        self.call_sid = start.get("callSid")
        media_format = start.get("mediaFormat") or {}
        self._log(
            "twilio.start",
            encoding=media_format.get("encoding"),
            sample_rate=media_format.get("sampleRate"),
            engines_ready=self._runtime.engines_ready,
            engine_error=self._runtime.engine_error,
        )
        if not self._runtime.engines_ready:
            self._log("engines.not_ready", error=self._runtime.engine_error)
        greeting = self._runtime.settings.greeting_text
        self._history.append({"role": "assistant", "content": greeting})
        self._turn_task = asyncio.create_task(self._speak(greeting), name="greeting")
        self._log("tts.greeting.queued", text=_preview(greeting))

    async def _on_media(self, data: dict[str, Any]) -> None:
        payload = (data.get("media") or {}).get("payload")
        if not payload:
            return
        pcm = mulaw_b64_to_pcm16(payload)
        self._media_in += 1
        if self._media_in == 1 or self._media_in % MEDIA_LOG_EVERY == 0:
            self._log("twilio.media.in", chunks=self._media_in, samples=pcm.size)
        if self._vad is None:
            self._utterance.append(pcm)
            return

        result = self._vad.process(pcm)
        if result.speech_started:
            self._log("vad.speech_started", probability=round(result.probability, 3))
            if self._assistant_speaking:
                await self._barge_in()
            self._utterance.append(self._pre_roll.flush())
        if result.is_speech or self._vad_in_speech():
            self._utterance.append(pcm)
            self._pre_roll.clear()
        else:
            self._pre_roll.append(pcm)
            if len(self._pre_roll) > self._runtime.settings.twilio_sample_rate:
                overflow = self._pre_roll.flush()
                self._pre_roll.append(overflow[-self._runtime.settings.twilio_sample_rate // 5 :])
        if result.speech_ended:
            utterance = self._utterance.flush()
            duration_ms = int(1000 * utterance.size / max(self._runtime.settings.twilio_sample_rate, 1))
            self._log("vad.speech_ended", samples=utterance.size, duration_ms=duration_ms)
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            self._turn_task = asyncio.create_task(self._run_turn(utterance), name="turn")

    def _vad_in_speech(self) -> bool:
        return bool(self._vad and getattr(self._vad, "in_speech", False))

    async def _barge_in(self) -> None:
        self._log("barge_in")
        self._cancel.set()
        self._assistant_speaking = False
        if self.stream_sid:
            await self._send_text(clear_stream(self.stream_sid))
        if self._turn_task and not self._turn_task.done() and self._turn_task is not asyncio.current_task():
            self._turn_task.cancel()

    async def _run_turn(self, pcm8k: np.ndarray) -> None:
        if pcm8k.size < MIN_UTTERANCE_SAMPLES_8K:
            self._log("vad.utterance_dropped", samples=pcm8k.size)
            return
        async with self._turn_lock:
            self._cancel.clear()
            turn_started = time.perf_counter()
            try:
                text = await self._transcribe(pcm8k)
                if self._cancel.is_set() or self._closed:
                    self._log("turn.cancelled", after="stt")
                    return
                if not text:
                    self._log("stt.empty")
                    return
                reply = await self._think(text)
                if self._cancel.is_set() or self._closed:
                    self._log("turn.cancelled", after="llm")
                    return
                if reply:
                    await self._speak(reply)
                self._log("turn.done", ms=_ms(turn_started))
            except asyncio.CancelledError:
                self._log("turn.cancelled")
                raise
            except Exception:
                logger.exception("step=turn.error stream=%s call=%s", self.stream_sid, self.call_sid)
                if not self._cancel.is_set() and not self._closed:
                    await self._speak(FALLBACK_REPLY)

    async def _transcribe(self, pcm8k: np.ndarray) -> str:
        stt = self._runtime.stt
        if stt is None:
            raise RuntimeError("STT non charge")
        pcm16k = resample_pcm16(
            pcm8k,
            self._runtime.settings.twilio_sample_rate,
            self._runtime.settings.stt_sample_rate,
        )
        self._log("stt.start", samples_8k=pcm8k.size, samples_16k=pcm16k.size)
        started = time.perf_counter()
        text = (await stt.transcribe_utterance(pcm16k)).strip()
        self._log("stt.done", ms=_ms(started), chars=len(text), text=_preview(text))
        return text

    async def _think(self, user_text: str) -> str:
        llm = self._runtime.llm
        if llm is None:
            raise RuntimeError("LLM non charge")
        self._history.append({"role": "user", "content": user_text})
        rounds = 0
        self._log("llm.start", history=len(self._history))
        started = time.perf_counter()
        while rounds < self._runtime.settings.llm_max_tool_rounds:
            if self._cancel.is_set():
                return ""
            result: LLMResult = await llm.complete(self._history, LLM_TOOLS)
            if result.tool_calls:
                self._history.append(_assistant_tool_message(result))
                for call in result.tool_calls:
                    self._log("llm.tool", name=call.name, call_id=call.id)
                    tool_started = time.perf_counter()
                    tool_result = await self._runtime.dispatcher.dispatch(call.name, call.arguments)
                    self._log(
                        "tool.done",
                        name=call.name,
                        ms=_ms(tool_started),
                        success=tool_result.get("success"),
                        error=tool_result.get("error"),
                        preview=_preview(json.dumps(tool_result, ensure_ascii=False), 300),
                    )
                    self._history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                rounds += 1
                continue
            content = (result.content or "").strip()
            self._log("llm.done", ms=_ms(started), rounds=rounds, chars=len(content), text=_preview(content))
            if content:
                self._history.append({"role": "assistant", "content": content})
                return content
            return FALLBACK_REPLY
        self._log("llm.max_rounds", rounds=rounds)
        return FALLBACK_REPLY

    async def _speak(self, text: str) -> None:
        tts = self._runtime.tts
        if tts is None:
            raise RuntimeError("TTS non charge")
        self._log("tts.start", chars=len(text), text=_preview(text))
        started = time.perf_counter()
        pcm8k = await tts.synthesize(text)
        if self._cancel.is_set() or self._closed or pcm8k.size == 0:
            self._log("tts.skipped", samples=pcm8k.size, cancelled=self._cancel.is_set(), closed=self._closed)
            return
        self._assistant_speaking = True
        frames_sent = 0
        try:
            mulaw = pcm16_to_mulaw(pcm8k)
            if not self.stream_sid or self._ws is None:
                self._log("tts.no_stream")
                return
            for frame in split_mulaw_frames(mulaw):
                if self._cancel.is_set() or self._closed:
                    self._log("tts.interrupted", frames=frames_sent)
                    break
                await self._send_text(media_out(self.stream_sid, _b64(frame)))
                frames_sent += 1
                self._media_out += 1
            self._log("tts.done", ms=_ms(started), samples=pcm8k.size, frames=frames_sent)
        finally:
            self._assistant_speaking = False

    async def _send_text(self, message: str) -> None:
        websocket = self._ws
        if websocket is None:
            return
        async with self._send_lock:
            if self._closed or self._ws is None:
                return
            try:
                await websocket.send_text(message)
            except Exception as exc:
                self._log("twilio.send_error", error=str(exc)[:200])


def _b64(frame: bytes) -> str:
    return base64.b64encode(frame).decode("ascii")


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _preview(value: str, limit: int = 200) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _assistant_tool_message(result: LLMResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in result.tool_calls
        ],
    }
