"""Session vocale isolee : buffer, VAD, tour de parole, barge-in minimal."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from starlette.websockets import WebSocketDisconnect

from app.audio.buffer import AudioBuffer
from app.audio.chunking import flush_speech_chunks
from app.audio.codec import (
    OUTBOUND_FRAME_SECONDS,
    mulaw_b64_to_pcm16,
    pcm16_to_mulaw,
    split_mulaw_frames,
)
from app.audio.preprocess import apply_preprocess
from app.audio.resample import resample_pcm16
from app.llm import LLMResult
from app.llm.tools import LLM_TOOLS
from app.node_client import NodeApiError
from app.observability import emit_event, sanitize_log_fields
from app.prompts import build_agent_instructions
from app.session.bootstrap import (
    apply_forced_caller_number,
    build_call_plan,
    normalize_french_caller_number,
    sanitize_identifier,
)
from app.session.interrupt import BargeInConfirmer, PlaybackEpoch
from app.session.messages import (
    apply_forced_caller_to_result,
    assistant_tool_message,
    stt_turns,
)
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
        self.instance_id: str | None = None
        self.stage = "connecting"
        self.last_error: str | None = None
        self._started_at = time.monotonic()
        self._caller_number: str | None = None
        self._history: list[dict[str, Any]] = [
            {"role": "system", "content": build_agent_instructions()},
        ]
        self._pre_roll = AudioBuffer()
        self._utterance = AudioBuffer()
        self._send_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._cancel = asyncio.Event()
        self._turn_task: asyncio.Task | None = None
        self._assistant_speaking = False
        self._closed = False
        self._live_create: dict[str, Any] | None = None
        self._media_in = 0
        self._media_out = 0
        self._vad = runtime.vad_model.new_session(runtime.settings) if runtime.vad_model else None
        self._context_ready = False
        self._pending_user = ""
        self._incomplete_task: asyncio.Task | None = None
        self._barge_gate = BargeInConfirmer()
        self._epoch = PlaybackEpoch()
        self._speech_started_at: float | None = None
        self._turn_ended_at: float | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def sanitized_snapshot(self) -> dict[str, Any]:
        return {
            "streamSid": sanitize_identifier(self.stream_sid),
            "callSid": sanitize_identifier(self.call_sid),
            "instanceId": self.instance_id,
            "stage": self.stage,
            "duration_seconds": round(max(0.0, time.monotonic() - self._started_at), 1),
            "provider": self._runtime.active_llm_provider,
            "last_error": self.last_error,
        }

    def _log(self, step: str, **fields: Any) -> None:
        parts = [f"step={step}", f"stream={self.stream_sid}", f"call={self.call_sid}"]
        for key, value in sanitize_log_fields(fields).items():
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
            self.stage = "connected"
            self._log("twilio.connected", protocol=data.get("protocol"), version=data.get("version"))
            return False
        if event == "start":
            return await self._on_start(data)
        if event == "media":
            await self._on_media(data)
            return False
        if event == "mark":
            mark = data.get("mark") or {}
            self._log("twilio.mark", name=mark.get("name"))
            return False
        if event in {"stop", "closed"}:
            self.stage = "stopping"
            self._log("twilio.stop", media_in=self._media_in, media_out=self._media_out)
            return True
        self._log("twilio.event", event=event)
        return False

    async def close(self, reason: str = "disconnect") -> None:
        if self._closed:
            self._log("session.close.idempotent", reason=reason)
            return
        self._closed = True
        self.stage = "closed"
        self._cancel.set()
        self._epoch.bump()
        self._barge_gate.reset()
        if self._incomplete_task and not self._incomplete_task.done():
            self._incomplete_task.cancel()
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
            except Exception as exc:
                # Dernier recours volontaire : le turn task a deja son propre
                # except Exception (voir _run_turn) qui avale et repond un
                # fallback vocal ; s'il leve quand meme (ex: le fallback lui-
                # meme echoue), la fermeture de session ne doit jamais planter.
                logger.error(
                    "step=session.task_error stream=%s error=%s",
                    self.stream_sid,
                    exc,
                    exc_info=True,
                )
        hangup_payload = self._build_hangup_payload()
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
        await self._report_hangup(hangup_payload)
        self._log(
            "session.close",
            reason=reason,
            task_cancelled=task_cancelled,
            buffered_samples=buffered,
            history=history_len,
            media_in=self._media_in,
            media_out=self._media_out,
            live_create=bool(self._live_create),
        )

    async def _on_start(self, data: dict[str, Any]) -> bool:
        plan = await build_call_plan(
            data,
            self._runtime.settings,
            self._runtime.node,
            self._runtime.telemetry,
        )
        start = data.get("start") or {}
        self.stream_sid = plan.stream_sid or None
        self.call_sid = plan.call_sid or None
        self._caller_number = plan.caller_number
        if plan.rejected:
            self.last_error = plan.last_error
            self.instance_id = None
            self.stage = "rejected"
            self._log("twilio.start.rejected", reason=plan.reject_reason)
            return True
        self.instance_id = plan.instance_id
        self._context_ready = plan.context_ready
        if plan.last_error:
            self.last_error = plan.last_error
            if plan.context_unavailable:
                self._log("voice.context.unavailable")
        self._history[0] = plan.seed_messages[0]
        self.stage = "greeting"
        media_format = start.get("mediaFormat") or {}
        self._log(
            "twilio.start",
            encoding=media_format.get("encoding"),
            sample_rate=media_format.get("sampleRate"),
            engines_ready=self._runtime.engines_ready,
            context_ready=self._context_ready,
        )
        if not self._runtime.engines_ready:
            self._log("engines.not_ready")
        self._history.append({"role": "assistant", "content": plan.greeting})
        self._turn_task = asyncio.create_task(self._speak(plan.greeting), name="greeting")
        self._log(
            "tts.greeting.queued",
            chars=len(plan.greeting),
            degraded=plan.context_unavailable,
        )
        return False

    async def _on_media(self, data: dict[str, Any]) -> None:
        payload = (data.get("media") or {}).get("payload")
        if not payload:
            return
        pcm = mulaw_b64_to_pcm16(payload)
        self._media_in += 1
        if self._media_in == 1 or self._media_in % MEDIA_LOG_EVERY == 0:
            self._log("twilio.media.in", chunks=self._media_in, samples=pcm.size)
        if self._vad is None:
            if not self._assistant_speaking:
                self._utterance.append(pcm)
            return

        vad_started = time.perf_counter()
        try:
            result = self._vad.process(pcm)
        except Exception:
            # VADEngine (app/vad) est un Protocol interchangeable : le type
            # d'exception concret depend du backend (ex: erreurs onnxruntime
            # pour Silero, qui ne partagent pas de base commune avec autre
            # chose qu'Exception). On tague puis on repropage tel quel ; le
            # traceback complet est logue plus haut par l'appelant
            # (main.py:_run_native_media_stream logue avec logger.exception).
            self.last_error = "vad_failed"
            if self._runtime.telemetry:
                self._runtime.telemetry.record_error("vad")
            raise
        finally:
            if self._runtime.telemetry:
                self._runtime.telemetry.record_latency("vad", _ms(vad_started))
        if result.speech_started:
            self._speech_started_at = time.perf_counter()
            self._log("vad.speech_started", probability=round(result.probability, 3))
            emit_event(
                "speech_start",
                call_id=self.call_sid,
                session_id=self.stream_sid,
                provider=self._runtime.settings.vad_provider,
            )
            self._utterance.append(self._pre_roll.flush())
        speaking = bool(result.is_speech or self._vad_in_speech())
        if speaking:
            self._utterance.append(pcm)
            self._pre_roll.clear()
            if self._assistant_speaking and self._runtime.settings.barge_in_enabled:
                confirm_ms = int(getattr(self._runtime.settings, "barge_in_confirm_ms", 160) or 0)
                if self._barge_gate.observe(
                    pcm.size,
                    True,
                    confirm_ms=confirm_ms,
                    sample_rate=self._runtime.settings.twilio_sample_rate,
                ):
                    await self._barge_in()
        else:
            self._barge_gate.reset()
            self._pre_roll.append(pcm)
            if len(self._pre_roll) > self._runtime.settings.twilio_sample_rate:
                overflow = self._pre_roll.flush()
                self._pre_roll.append(overflow[-self._runtime.settings.twilio_sample_rate // 5 :])
        if result.speech_ended:
            self._barge_gate.reset()
            utterance = self._utterance.flush()
            self._turn_ended_at = time.perf_counter()
            duration_ms = int(1000 * utterance.size / max(self._runtime.settings.twilio_sample_rate, 1))
            self._log("vad.speech_ended", samples=utterance.size, duration_ms=duration_ms)
            emit_event(
                "speech_end",
                call_id=self.call_sid,
                session_id=self.stream_sid,
                latency_ms=duration_ms,
            )
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            self._turn_task = asyncio.create_task(self._run_turn(utterance), name="turn")

    def _vad_in_speech(self) -> bool:
        return bool(self._vad and getattr(self._vad, "in_speech", False))

    def _playback_live(self, generation: int) -> bool:
        return self._epoch.is_current(generation) and not self._cancel.is_set() and not self._closed

    async def _barge_in(self) -> None:
        self._log("barge_in", generation=self._epoch.current)
        self._epoch.bump()
        self._barge_gate.reset()
        emit_event(
            "interruption_detected",
            call_id=self.call_sid,
            session_id=self.stream_sid,
            provider=self._runtime.settings.vad_provider,
            generation=self._epoch.current,
        )
        self._cancel.set()
        self._assistant_speaking = False
        if self.stream_sid:
            await self._send_text(clear_stream(self.stream_sid))
            emit_event(
                "audio_buffer_cleared",
                call_id=self.call_sid,
                session_id=self.stream_sid,
            )
        if self._turn_task and not self._turn_task.done() and self._turn_task is not asyncio.current_task():
            self._turn_task.cancel()

    async def _run_turn(self, pcm8k: np.ndarray) -> None:
        if pcm8k.size < MIN_UTTERANCE_SAMPLES_8K:
            self._log("vad.utterance_dropped", samples=pcm8k.size)
            return
        async with self._turn_lock:
            self._cancel.clear()
            self.stage = "processing"
            previous_error = self.last_error
            turn_started = time.perf_counter()
            try:
                text = await self._transcribe(pcm8k)
                if self._cancel.is_set() or self._closed:
                    self._log("turn.cancelled", after="stt")
                    return
                if not text:
                    self._log("stt.empty")
                    return
                combined = f"{self._pending_user} {text}".strip()
                detector = self._runtime.turn
                if detector is not None and not detector.is_complete(combined):
                    self._pending_user = combined
                    emit_event(
                        "turn_end_detected",
                        call_id=self.call_sid,
                        session_id=self.stream_sid,
                        provider=getattr(detector, "name", "unknown"),
                        complete=False,
                    )
                    self._schedule_incomplete_flush()
                    return
                self._pending_user = ""
                emit_event(
                    "turn_end_detected",
                    call_id=self.call_sid,
                    session_id=self.stream_sid,
                    provider=getattr(detector, "name", "unknown"),
                    complete=True,
                )
                await self._respond(combined)
                self._log("turn.done", ms=_ms(turn_started))
            except asyncio.CancelledError:
                self._log("turn.cancelled")
                raise
            except Exception as exc:
                # Dernier recours volontaire et assume : STT/LLM/TTS/outils
                # sont des moteurs interchangeables (Protocol) dont les types
                # d'exception concrets varient par backend ; ce tour ne doit
                # jamais planter la session, donc on avale et on repond un
                # fallback vocal. On logge le type/traceback ici (seul point
                # ou l'exception est visible) pour ne pas perdre le diagnostic
                # reseau/timeout/bug que le fallback masquerait sinon.
                if self.last_error == previous_error:
                    self.last_error = "turn_failed"
                    if self._runtime.telemetry:
                        self._runtime.telemetry.record_error("turn")
                logger.error(
                    "step=turn.error stream=%s call=%s error=%s",
                    self.stream_sid,
                    self.call_sid,
                    exc,
                    exc_info=True,
                )
                if not self._cancel.is_set() and not self._closed:
                    await self._speak(FALLBACK_REPLY)

    def _schedule_incomplete_flush(self) -> None:
        if self._incomplete_task and not self._incomplete_task.done():
            self._incomplete_task.cancel()
        wait_s = max(0.1, self._runtime.settings.turn_incomplete_wait_ms / 1000)
        self._incomplete_task = asyncio.create_task(self._flush_incomplete(wait_s), name="incomplete")

    async def _flush_incomplete(self, wait_s: float) -> None:
        await asyncio.sleep(wait_s)
        text = self._pending_user
        if not text or self._closed:
            return
        self._pending_user = ""
        async with self._turn_lock:
            await self._respond(text)

    async def _respond(self, user_text: str) -> None:
        if self._runtime.settings.llm_stream and callable(getattr(self._runtime.llm, "complete_stream", None)):
            await self._stream_or_think(user_text)
            return
        reply = await self._think(user_text)
        if self._cancel.is_set() or self._closed:
            self._log("turn.cancelled", after="llm")
            return
        if reply:
            await self._speak(reply)

    async def _stream_or_think(self, user_text: str) -> None:
        llm = self._runtime.llm
        if llm is None:
            raise RuntimeError("LLM non charge")
        stream = llm.complete_stream
        self._history.append({"role": "user", "content": user_text})
        tools = LLM_TOOLS if self._context_ready else []
        buffer = ""
        spoken: list[str] = []
        saw_tools = False
        emit_event(
            "llm_start",
            call_id=self.call_sid,
            session_id=self.stream_sid,
            provider=self._runtime.active_llm_provider,
            model=self._runtime.settings.vllm_model,
        )
        llm_started = time.perf_counter()
        first_token_logged = False
        async for delta in stream(self._history, tools):
            if self._cancel.is_set() or self._closed:
                return
            if delta.tool_calls:
                saw_tools = True
                break
            if not delta.content:
                continue
            if not first_token_logged:
                first_token_logged = True
                ttft_ms = _ms(llm_started)
                if self._runtime.telemetry:
                    self._runtime.telemetry.record_latency("llm_ttft", ttft_ms)
                emit_event(
                    "llm_first_token",
                    call_id=self.call_sid,
                    session_id=self.stream_sid,
                    latency_ms=ttft_ms,
                    provider=self._runtime.active_llm_provider,
                )
            buffer += delta.content
            ready, buffer = flush_speech_chunks(buffer, self._runtime.settings.tts_chunking)
            for chunk in ready:
                spoken.append(chunk)
                await self._speak(chunk)
        if saw_tools:
            self._history.pop()
            reply = await self._think(user_text)
            if reply and not self._cancel.is_set() and not self._closed:
                await self._speak(reply)
            return
        remainder = buffer.strip()
        if remainder:
            spoken.append(remainder)
            await self._speak(remainder)
        full = " ".join(spoken).strip()
        emit_event(
            "llm_complete",
            call_id=self.call_sid,
            session_id=self.stream_sid,
            provider=self._runtime.active_llm_provider,
        )
        if full:
            self._history.append({"role": "assistant", "content": full})
            return
        await self._speak(FALLBACK_REPLY)

    async def _transcribe(self, pcm8k: np.ndarray) -> str:
        stt = self._runtime.stt
        if stt is None:
            raise RuntimeError("STT non charge")
        prepared = apply_preprocess(pcm8k, self._runtime.settings.audio_preprocess)
        pcm16k = resample_pcm16(
            prepared,
            self._runtime.settings.twilio_sample_rate,
            self._runtime.settings.stt_sample_rate,
        )
        self.stage = "stt"
        self._log("stt.start", samples_8k=prepared.size, samples_16k=pcm16k.size)
        if self._speech_started_at is not None and self._runtime.telemetry:
            audio_to_stt_ms = _ms(self._speech_started_at)
            self._runtime.telemetry.record_latency("audio_in_to_stt_start", audio_to_stt_ms)
            emit_event(
                "audio_in_to_stt_start",
                call_id=self.call_sid,
                session_id=self.stream_sid,
                latency_ms=audio_to_stt_ms,
            )
            self._speech_started_at = None
        emit_event(
            "stt_start",
            call_id=self.call_sid,
            session_id=self.stream_sid,
            provider=self._runtime.stt_provider,
            model=self._runtime.settings.whisper_model_size,
        )
        started = time.perf_counter()
        try:
            text = (await stt.transcribe_utterance(pcm16k)).strip()
            self._log("stt.done", ms=_ms(started), chars=len(text))
            emit_event(
                "stt_final",
                call_id=self.call_sid,
                session_id=self.stream_sid,
                provider=self._runtime.stt_provider,
                latency_ms=_ms(started),
            )
            return text
        except Exception:
            # STTEngine est un Protocol interchangeable (faster-whisper,
            # parakeet/NeMo...) ; on tague puis repropage sans filtrer le
            # type, le traceback est logue au niveau orchestrateur du tour
            # (_run_turn) qui est le seul point qui avale l'exception.
            self.last_error = "stt_failed"
            if self._runtime.telemetry:
                self._runtime.telemetry.record_error("stt")
            raise
        finally:
            if self._runtime.telemetry:
                self._runtime.telemetry.record_latency("stt", _ms(started))

    async def _think(self, user_text: str) -> str:
        llm = self._runtime.llm
        if llm is None:
            raise RuntimeError("LLM non charge")
        self._history.append({"role": "user", "content": user_text})
        rounds = 0
        self.stage = "llm"
        self._log("llm.start", history=len(self._history))
        emit_event(
            "llm_start",
            call_id=self.call_sid,
            session_id=self.stream_sid,
            provider=self._runtime.active_llm_provider,
            model=self._runtime.settings.vllm_model,
        )
        started = time.perf_counter()
        try:
            while rounds < self._runtime.settings.llm_max_tool_rounds:
                if self._cancel.is_set():
                    return ""
                round_started = time.perf_counter()
                tools = LLM_TOOLS if self._context_ready else []
                result: LLMResult = await llm.complete(self._history, tools)
                await self._report_usage(result, _ms(round_started))
                apply_forced_caller_to_result(result, self._caller_number)
                if result.tool_calls:
                    self._history.append(assistant_tool_message(result))
                    for call in result.tool_calls:
                        self.stage = f"tool:{call.name}"
                        self._log("llm.tool", name=call.name, call_id=call.id)
                        emit_event(
                            "tool_start",
                            call_id=self.call_sid,
                            session_id=self.stream_sid,
                            provider="fastify",
                            name=call.name,
                        )
                        tool_started = time.perf_counter()
                        tool_result = await self._runtime.dispatcher.dispatch(
                            call.name,
                            call.arguments,
                            self.instance_id,
                        )
                        self._remember_live_create(call.name, tool_result)
                        self._log(
                            "tool.done",
                            name=call.name,
                            ms=_ms(tool_started),
                            success=tool_result.get("success"),
                        )
                        emit_event(
                            "tool_complete",
                            call_id=self.call_sid,
                            session_id=self.stream_sid,
                            latency_ms=_ms(tool_started),
                            name=call.name,
                        )
                        self._history.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": json.dumps(tool_result, ensure_ascii=False),
                            }
                        )
                    rounds += 1
                    self.stage = "llm"
                    continue
                content = (result.content or "").strip()
                self._log("llm.done", ms=_ms(started), rounds=rounds, chars=len(content))
                if content:
                    self._history.append({"role": "assistant", "content": content})
                    return content
                return FALLBACK_REPLY
            self._log("llm.max_rounds", rounds=rounds)
            return FALLBACK_REPLY
        except Exception:
            # LLMProvider est un Protocol interchangeable (vLLM/OpenAI) et
            # cette boucle appelle aussi le dispatcher d'outils Fastify (deja
            # convertis en NodeApiError -> dict conversationnel, donc pas
            # d'exception attendue de ce cote). On tague puis repropage ; le
            # traceback est logue au niveau orchestrateur du tour (_run_turn).
            self.last_error = "llm_failed"
            if self._runtime.telemetry:
                self._runtime.telemetry.record_error("llm")
            raise
        finally:
            if self._runtime.telemetry:
                self._runtime.telemetry.record_latency("llm", _ms(started))

    async def _speak(self, text: str) -> None:
        tts = self._runtime.tts
        if tts is None:
            raise RuntimeError("TTS non charge")
        self.stage = "tts"
        self._log("tts.start", chars=len(text))
        emit_event(
            "tts_start",
            call_id=self.call_sid,
            session_id=self.stream_sid,
            provider=self._runtime.tts_provider,
            model=self._runtime.settings.kokoro_voice,
        )
        started = time.perf_counter()
        first_audio = False
        generation = self._epoch.current
        try:
            stream = getattr(tts, "synthesize_stream", None)
            if self._runtime.settings.tts_stream and callable(stream):
                async for pcm8k in stream(text):
                    first_audio = await self._play_pcm(pcm8k, started, first_audio, generation)
            else:
                pcm8k = await tts.synthesize(text)
                first_audio = await self._play_pcm(pcm8k, started, first_audio, generation)
            emit_event(
                "tts_complete",
                call_id=self.call_sid,
                session_id=self.stream_sid,
                latency_ms=_ms(started),
                provider=self._runtime.tts_provider,
            )
        except Exception:
            # TTSEngine est un Protocol interchangeable (Kokoro aujourd'hui) ;
            # on tague puis repropage sans filtrer le type, le traceback est
            # logue au niveau orchestrateur du tour (_run_turn) ou par le
            # WebSocket handler si l'appel vient de _speak(greeting) hors tour
            # (voir main.py:_run_native_media_stream, logger.exception).
            self.last_error = "tts_failed"
            if self._runtime.telemetry:
                self._runtime.telemetry.record_error("tts")
            raise
        finally:
            self._assistant_speaking = False
            if not self._closed:
                self.stage = "listening"
            if self._runtime.telemetry:
                self._runtime.telemetry.record_latency("tts", _ms(started))

    async def _play_pcm(
        self,
        pcm8k: np.ndarray,
        started: float,
        first_audio: bool,
        generation: int,
    ) -> bool:
        if not self._playback_live(generation) or pcm8k.size == 0:
            self._log("tts.skipped", samples=pcm8k.size, cancelled=self._cancel.is_set(), closed=self._closed)
            return first_audio
        self._assistant_speaking = True
        mulaw = pcm16_to_mulaw(pcm8k)
        if not self.stream_sid or self._ws is None:
            self._log("tts.no_stream")
            return first_audio
        frames = split_mulaw_frames(mulaw)
        next_frame_at = time.monotonic()
        frames_sent = 0
        for index, frame in enumerate(frames):
            if not self._playback_live(generation):
                self._log("tts.interrupted", frames=frames_sent, generation=generation)
                break
            if index > 0:
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            if not self._playback_live(generation):
                self._log("tts.interrupted", frames=frames_sent, generation=generation)
                break
            await self._send_text(media_out(self.stream_sid, _b64(frame)))
            frames_sent += 1
            self._media_out += 1
            if not first_audio:
                first_audio = True
                ttfa_ms = _ms(started)
                if self._runtime.telemetry:
                    # Le moteur est construit avant l'appel : cette mesure ne
                    # comprend ni le chargement ni le telechargement du modele.
                    self._runtime.telemetry.record_latency("tts_ttfa_excluding_load", ttfa_ms)
                emit_event(
                    "tts_first_audio",
                    call_id=self.call_sid,
                    session_id=self.stream_sid,
                    latency_ms=ttfa_ms,
                    provider=self._runtime.tts_provider,
                )
                if self._turn_ended_at is not None:
                    end_to_audio_ms = _ms(self._turn_ended_at)
                    if self._runtime.telemetry:
                        self._runtime.telemetry.record_latency(
                            "end_turn_to_first_audio", end_to_audio_ms
                        )
                    emit_event(
                        "end_turn_to_first_audio",
                        call_id=self.call_sid,
                        session_id=self.stream_sid,
                        latency_ms=end_to_audio_ms,
                    )
                    self._turn_ended_at = None
                emit_event(
                    "twilio_first_audio_sent",
                    call_id=self.call_sid,
                    session_id=self.stream_sid,
                    latency_ms=ttfa_ms,
                )
            next_frame_at += OUTBOUND_FRAME_SECONDS
        self._log("tts.done", ms=_ms(started), samples=pcm8k.size, frames=frames_sent)
        return first_audio

    async def _report_usage(self, result: LLMResult, latency_ms: int) -> None:
        if not self.instance_id:
            return
        if (result.input_tokens or 0) + (result.output_tokens or 0) <= 0:
            return
        provider = str(getattr(self._runtime.llm, "active_provider", None) or "vllm")
        if provider not in {"vllm", "openai"}:
            provider = "vllm"
        try:
            await self._runtime.node.report_usage(
                {
                    "instanceId": self.instance_id,
                    "source": "voice",
                    "provider": provider,
                    "model": result.model or "",
                    "inputTokens": int(result.input_tokens or 0),
                    "outputTokens": int(result.output_tokens or 0),
                    "totalTokens": int(result.total_tokens or 0),
                    "latencyMs": int(latency_ms or 0),
                }
            )
        except NodeApiError as exc:
            # Le reporting d'usage ne doit jamais casser le tour en cours ;
            # NodeApiClient convertit deja les pannes reseau/timeout httpx en
            # NodeApiError (voir app/node_client.py), donc c'est le seul type
            # attendu ici.
            logger.warning("step=llm.usage_report_failed error=%s", str(exc)[:200])

    def _remember_live_create(self, name: str, tool_result: dict[str, Any]) -> None:
        if name != "create_appointment":
            return
        if not tool_result.get("success"):
            return
        created_type = tool_result.get("type")
        created_id = tool_result.get("id")
        live: dict[str, Any] = {}
        if created_type in {"reservation", "order"}:
            live["type"] = created_type
        if created_id is not None:
            text_id = str(created_id).strip()
            if text_id:
                live["id"] = text_id
        self._live_create = live or None

    def _build_hangup_payload(self) -> dict[str, Any] | None:
        if not self.instance_id:
            return None
        payload: dict[str, Any] = {
            "instanceId": self.instance_id,
            "streamSid": self.stream_sid,
            "callSid": self.call_sid,
            "durationSeconds": round(max(0.0, time.monotonic() - self._started_at), 1),
            "turns": stt_turns(self._history),
            "liveCreate": self._live_create,
        }
        return payload

    async def _report_hangup(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        report = getattr(self._runtime.node, "report_call_ended", None)
        if not callable(report):
            return
        try:
            await report(payload, tenant_id=self.instance_id)
            self._log(
                "hangup.reported",
                turns=len(payload.get("turns") or []),
                live_create=bool(payload.get("liveCreate")),
                created_type=(payload.get("liveCreate") or {}).get("type"),
            )
        except NodeApiError as exc:
            # Meme raisonnement que _report_usage : NodeApiClient convertit
            # deja reseau/timeout en NodeApiError, la fermeture de session ne
            # doit pas echouer si Fastify est injoignable au hangup.
            logger.warning("step=hangup.failed error=%s", str(exc)[:200])

    async def _send_text(self, message: str) -> None:
        websocket = self._ws
        if websocket is None:
            return
        async with self._send_lock:
            if self._closed or self._ws is None:
                return
            try:
                await websocket.send_text(message)
            except (RuntimeError, WebSocketDisconnect) as exc:
                # Seuls types que Starlette leve depuis WebSocket.send() : un
                # OSError reseau est deja converti en WebSocketDisconnect en
                # interne, et RuntimeError couvre un envoi apres fermeture.
                self._log("twilio.send_error", error=str(exc)[:200])


def _b64(frame: bytes) -> str:
    return base64.b64encode(frame).decode("ascii")


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
