"""A/B Native vs Pipecat : orchestration, pas latence STT CPU."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from app.ab.doubles import (
    AlwaysSpeechVad,
    CycleVad,
    CycleVadModel,
    FakeWs,
    InstantTTS,
    ProbeLLM,
    QuietVad,
    SequenceSTT,
    SlowTTS,
    TrackingNode,
)
from app.audio.codec import pcm16_to_mulaw
from app.audio.resample import resample_pcm16
from app.config import Settings
from app.pipeline.pipecat_runner import (
    _register_fastify_tools,
    _report_hangup,
    is_pipecat_available,
    run_pipecat_session,
)
from app.runtime import VoiceRuntime
from app.session.bootstrap import apply_forced_caller_number, build_call_plan
from app.session.call import CallSession
from app.stream_token import create_stream_token
from app.tools.dispatcher import ToolDispatcher
from app.turn.detector import HeuristicTurnDetector


REPEATS = 5
SECRET = "ab-stream-secret"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "audio"
HOLD_MS = 800
UNMEASURED_TWILIO = "not measured: live Twilio+Whisper pipeline not started"


def _p50(values: list[float]) -> float | None:
    return round(median(values), 2) if values else None


def _media(pcm: np.ndarray | None = None) -> str:
    samples = pcm if pcm is not None else np.zeros(2000, dtype=np.int16)
    payload = base64.b64encode(pcm16_to_mulaw(samples)).decode("ascii")
    return json.dumps({"event": "media", "media": {"payload": payload}})


def _start(instance: bool = False, caller: str | None = None) -> str:
    start: dict[str, Any] = {"streamSid": "MZ1", "callSid": "CA1"}
    if instance:
        custom: dict[str, Any] = {
            "instanceId": "restaurant-42",
            "streamToken": create_stream_token("restaurant-42", SECRET),
        }
        if caller:
            custom["callerNumber"] = caller
        start["customParameters"] = custom
    return json.dumps({"event": "start", "start": start})


def _runtime(node, llm, stt, tts, vad_session, hold_ms: int = HOLD_MS) -> VoiceRuntime:
    settings = Settings(
        voice_eager_load=False,
        greeting_text="Bonjour, je vous ecoute.",
        twilio_auth_token=SECRET,
        turn_incomplete_wait_ms=hold_ms,
        barge_in_enabled=True,
        barge_in_confirm_ms=160,
        llm_stream=False,
        tts_stream=False,
    )
    return VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=llm,
        stt=stt,
        tts=tts,
        vad_model=CycleVadModel(vad_session),
        engines_ready=True,
        turn=HeuristicTurnDetector(),
    )


@contextmanager
def _patched_events(store: list[dict[str, Any]]):
    import app.session.call as call_mod

    original = call_mod.emit_event

    def probe(step: str, **fields: Any) -> None:
        store.append({"step": step, "t": time.perf_counter(), **fields})
        original(step, **fields)

    call_mod.emit_event = probe
    try:
        yield store
    finally:
        call_mod.emit_event = original


def _delta_ms(store: list[dict[str, Any]], start: str, end: str) -> float | None:
    starts = [item["t"] for item in store if item["step"] == start]
    ends = [item["t"] for item in store if item["step"] == end]
    if not starts or not ends:
        return None
    return (ends[-1] - starts[-1]) * 1000


def _media_after_last_clear(sent: list[str]) -> int:
    after = 0
    seen_clear = False
    for item in sent:
        if '"event": "clear"' in item or '"event":"clear"' in item:
            seen_clear = True
            after = 0
            continue
        if seen_clear and ('"event": "media"' in item or '"event":"media"' in item):
            after += 1
    return after


async def _greet(session: CallSession) -> None:
    await session.handle_raw(_start())
    if session._turn_task:
        await session._turn_task


async def _utter(session: CallSession, pcm: np.ndarray | None = None) -> None:
    await session.handle_raw(_media(pcm))
    await session.handle_raw(_media(pcm))
    if session._turn_task:
        await asyncio.wait_for(session._turn_task, timeout=2)


async def measure_native_turn(repeats: int = REPEATS) -> dict[str, Any]:
    immediate_false = 0
    held_then_flush = 0
    combined_ok = 0
    premature = 0
    late_clause_after_hold = 0
    hesitation_ok = 0
    correction_ok = 0
    short_dropped = 0
    eot_ms: list[float] = []
    complete_to_llm_ms: list[float] = []
    incomplete_to_llm_ms: list[float] = []

    for _ in range(repeats):
        llm = ProbeLLM()
        store: list[dict[str, Any]] = []
        with _patched_events(store):
            session = CallSession(
                FakeWs(),
                _runtime(TrackingNode(), llm, SequenceSTT(["Je voudrais reserver pour"]), InstantTTS(), CycleVad()),
            )
            await _greet(session)
            await _utter(session)
            if llm.calls == 0:
                await asyncio.sleep((HOLD_MS / 1000) + 0.15)
                if llm.calls >= 1:
                    held_then_flush += 1
                    delta = _delta_ms(store, "speech_end", "llm_start")
                    if delta is not None:
                        incomplete_to_llm_ms.append(delta)
            else:
                immediate_false += 1
            decision = _delta_ms(store, "speech_end", "turn_end_detected")
            if decision is not None:
                eot_ms.append(decision)
            await session.close()

        llm = ProbeLLM()
        stt = SequenceSTT(["Je voudrais reserver pour", "six personnes."])
        vad = CycleVad()
        session = CallSession(FakeWs(), _runtime(TrackingNode(), llm, stt, InstantTTS(), vad))
        await _greet(session)
        await _utter(session)
        first_calls = llm.calls
        vad.n = 0
        await _utter(session)
        if first_calls == 0 and llm.calls == 1:
            combined_ok += 1
        if first_calls > 0:
            premature += 1
        await session.close()

        llm = ProbeLLM()
        stt = SequenceSTT(["Je voudrais reserver pour", "six personnes."])
        session = CallSession(FakeWs(), _runtime(TrackingNode(), llm, stt, InstantTTS(), CycleVad()))
        await _greet(session)
        await _utter(session)
        await asyncio.sleep((HOLD_MS / 1000) + 0.15)
        calls_after_hold = llm.calls
        await _utter(session)
        if calls_after_hold >= 1:
            late_clause_after_hold += 1
        await session.close()

        llm = ProbeLLM()
        stt = SequenceSTT(["euh", "six personnes."])
        vad = CycleVad()
        session = CallSession(FakeWs(), _runtime(TrackingNode(), llm, stt, InstantTTS(), vad))
        await _greet(session)
        await _utter(session)
        first = llm.calls
        vad.n = 0
        await _utter(session)
        if first == 0 and llm.calls == 1:
            hesitation_ok += 1
        await session.close()

        llm = ProbeLLM()
        stt = SequenceSTT(["Attendez", "non, plutot vingt et une heures."])
        vad = CycleVad()
        session = CallSession(FakeWs(), _runtime(TrackingNode(), llm, stt, InstantTTS(), vad))
        await _greet(session)
        await _utter(session)
        first = llm.calls
        vad.n = 0
        await _utter(session)
        if first == 0 and llm.calls == 1:
            correction_ok += 1
        await session.close()

        llm = ProbeLLM()
        session = CallSession(FakeWs(), _runtime(TrackingNode(), llm, SequenceSTT(["x"]), InstantTTS(), CycleVad()))
        await _greet(session)
        await session.handle_raw(_media(np.zeros(400, dtype=np.int16)))
        await session.handle_raw(_media(np.zeros(400, dtype=np.int16)))
        if session._turn_task:
            await asyncio.wait_for(session._turn_task, timeout=2)
        if llm.calls == 0:
            short_dropped += 1
        await session.close()

        llm = ProbeLLM()
        store = []
        with _patched_events(store):
            session = CallSession(
                FakeWs(),
                _runtime(TrackingNode(), llm, SequenceSTT(["Oui."]), InstantTTS(), CycleVad()),
            )
            await _greet(session)
            await _utter(session)
            delta = _delta_ms(store, "speech_end", "llm_start")
            if delta is not None:
                complete_to_llm_ms.append(delta)
            await session.close()

    return {
        "repeats": repeats,
        "immediate_false_eot": immediate_false,
        "held_then_flush_false_eot": held_then_flush,
        "combined_before_hold": combined_ok,
        "premature_before_second_clause": premature,
        "late_clause_after_hold_already_answered": late_clause_after_hold,
        "hesitation_held_then_combined": hesitation_ok,
        "correction_held_then_combined": correction_ok,
        "short_utterance_dropped": short_dropped,
        "eot_decision_p50_ms": _p50(eot_ms),
        "complete_speech_end_to_llm_p50_ms": _p50(complete_to_llm_ms),
        "incomplete_speech_end_to_llm_p50_ms": _p50(incomplete_to_llm_ms),
        "hold_ms": HOLD_MS,
        "note": "FakeSTT/LLM/TTS. 800 ms is the configured hold, not an optimum.",
    }


async def measure_native_barge_in(repeats: int = REPEATS) -> dict[str, Any]:
    detect: list[float] = []
    clear: list[float] = []
    leftover: list[int] = []
    missing_clear = 0
    leftover_speaking = 0
    timing_failures = 0
    for _ in range(repeats):
        store: list[dict[str, Any]] = []
        ws = FakeWs()
        with _patched_events(store):
            session = CallSession(
                ws,
                _runtime(TrackingNode(), ProbeLLM(), SequenceSTT(["non"]), SlowTTS(), AlwaysSpeechVad()),
            )
            await _greet(session)
            speak = asyncio.create_task(
                session._speak("Tres bien, j'ai une table disponible a vingt heures.")
            )
            await asyncio.sleep(0.04)
            await session.handle_raw(_media(np.full(2000, 12000, dtype=np.int16)))
            await asyncio.wait_for(speak, timeout=3)
            d_detect = _delta_ms(store, "speech_start", "interruption_detected")
            d_clear = _delta_ms(store, "speech_start", "audio_buffer_cleared")
            if d_detect is None or d_clear is None:
                timing_failures += 1
            else:
                detect.append(d_detect)
                clear.append(d_clear)
            if session._assistant_speaking:
                leftover_speaking += 1
            if not any('"event": "clear"' in item or '"event":"clear"' in item for item in ws.sent):
                missing_clear += 1
            leftover.append(_media_after_last_clear(ws.sent))
            await session.close()
    return {
        "repeats": repeats,
        "barge_in_detect_p50_ms": _p50(detect),
        "buffer_clear_p50_ms": _p50(clear),
        "media_frames_after_last_clear_p50": _p50([float(v) for v in leftover]),
        "leftover_speaking": leftover_speaking,
        "missing_clear": missing_clear,
        "timing_failures": timing_failures,
    }


async def measure_native_false_interrupt(repeats: int = REPEATS) -> dict[str, Any]:
    silence_hits = 0
    low_amp_hits = 0
    short_burst_hits = 0
    noise_hits = 0
    rng = np.random.default_rng(0)
    for _ in range(repeats):
        async def _interrupt(pcm: np.ndarray, vad) -> bool:
            store: list[dict[str, Any]] = []
            with _patched_events(store):
                session = CallSession(
                    FakeWs(),
                    _runtime(TrackingNode(), ProbeLLM(), SequenceSTT(["x"]), SlowTTS(), vad),
                )
                await _greet(session)
                speak = asyncio.create_task(session._speak("J'ai une table disponible a vingt heures."))
                await asyncio.sleep(0.04)
                await session.handle_raw(_media(pcm))
                await asyncio.wait_for(speak, timeout=3)
                hit = any(item["step"] == "interruption_detected" for item in store)
                await session.close()
                return hit

        if await _interrupt(np.zeros(400, dtype=np.int16), QuietVad()):
            silence_hits += 1
        if await _interrupt(np.full(800, 40, dtype=np.int16), QuietVad()):
            low_amp_hits += 1
        if await _interrupt(np.full(400, 12000, dtype=np.int16), QuietVad()):
            short_burst_hits += 1
        noise = np.clip(rng.normal(0, 400, 2000), -32768, 32767).astype(np.int16)
        if await _interrupt(noise, QuietVad()):
            noise_hits += 1
    return {
        "repeats": repeats,
        "silence": silence_hits,
        "low_amplitude": low_amp_hits,
        "short_burst_50ms": short_burst_hits,
        "gaussian_noise_std400": noise_hits,
        "souffle_humain": UNMEASURED_TWILIO,
        "bruit_restaurant_reel": UNMEASURED_TWILIO,
        "echo_bot": UNMEASURED_TWILIO,
    }


async def measure_native_cleanup(repeats: int = REPEATS) -> dict[str, Any]:
    failures = 0
    double_hangup = 0
    leftover_tasks = 0
    for _ in range(repeats):
        node = TrackingNode()
        session = CallSession(
            FakeWs(),
            _runtime(node, ProbeLLM(), SequenceSTT(["ok"]), InstantTTS(), CycleVad()),
        )
        await session.handle_raw(_start(instance=True))
        if session._turn_task:
            await session._turn_task
        await session.close()
        await session.close()
        await asyncio.sleep(0.02)
        if not session.closed or session._ws is not None or len(session._history) != 0:
            failures += 1
        if len(node.hangups) != 1:
            double_hangup += 1
        unfinished = session._turn_task is not None and not session._turn_task.done()
        incomplete = session._incomplete_task is not None and not session._incomplete_task.done()
        if unfinished or incomplete:
            leftover_tasks += 1
    return {
        "repeats": repeats,
        "cleanup_failures": failures,
        "double_hangup": double_hangup,
        "leftover_tasks": leftover_tasks,
    }


async def measure_native_tools() -> dict[str, Any]:
    node = TrackingNode()
    session = CallSession(
        FakeWs(),
        _runtime(node, ProbeLLM(), SequenceSTT(["ok"]), InstantTTS(), CycleVad()),
    )
    await session.handle_raw(_start(instance=True, caller="+33612345678"))
    if session._turn_task:
        await session._turn_task
    avail = await session._runtime.dispatcher.dispatch(
        "check_availability", {"date": "2026-09-20"}, "restaurant-42"
    )
    created = await session._runtime.dispatcher.dispatch(
        "create_appointment",
        {
            "name": "Alice",
            "telephone": "0000000000",
            "date": "2026-09-20",
            "time": "20:00",
            "type": "Réservation de table",
            "nombrePersonnes": 4,
        },
        "restaurant-42",
    )
    forced = apply_forced_caller_number({"telephone": "0000000000"}, session._caller_number)
    update = await session._runtime.dispatcher.dispatch("update_appointment", {}, "restaurant-42")
    cancel = await session._runtime.dispatcher.dispatch("cancel_appointment", {}, "restaurant-42")
    await session.close()
    return {
        "check_availability_ok": bool(avail.get("success")),
        "create_appointment_ok": bool(created.get("success")),
        "forced_caller": forced.get("telephone") == "0612345678",
        "update_appointment": update,
        "cancel_appointment": cancel,
        "note": "LLM_TOOLS exposes only check_availability and create_appointment on both orchestrators.",
    }


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
        return np.frombuffer(frames, dtype=np.int16), rate


def _resolve_audio(row: dict[str, Any]) -> Path | None:
    path = AUDIO_ROOT.parent / row["path"]
    if path.exists():
        return path
    fallback = AUDIO_ROOT / Path(row["path"]).name
    return fallback if fallback.exists() else None


async def measure_smart_turn() -> dict[str, Any]:
    if not is_pipecat_available():
        return {"available": False, "full_twilio_pipeline": UNMEASURED_TWILIO}
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    manifest_path = AUDIO_ROOT / "manifest.json"
    if not manifest_path.exists():
        return {"available": True, "corpus": False, "full_twilio_pipeline": UNMEASURED_TWILIO}
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean = [row for row in rows if row.get("condition") == "clean"]
    street = [row for row in rows if row.get("condition") == "street"]
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000)
    analyzer.set_sample_rate(16000)

    async def decide(pcm16: np.ndarray, rate: int, fraction: float = 1.0) -> tuple[str, float, float | None]:
        audio = resample_pcm16(pcm16, rate, 16000)
        if fraction < 1.0:
            audio = audio[: max(1600, int(audio.size * fraction))]
        analyzer.clear()
        frame = 320
        for index in range(0, audio.size, frame):
            chunk = audio[index : index + frame]
            if chunk.size < frame:
                chunk = np.pad(chunk, (0, frame - chunk.size))
            analyzer.append_audio(chunk.tobytes(), True)
        analyzer.append_audio(np.zeros(frame, dtype=np.int16).tobytes(), False)
        started = time.perf_counter()
        state, metrics = await analyzer.analyze_end_of_turn()
        elapsed = (time.perf_counter() - started) * 1000
        probability = getattr(metrics, "probability", None) if metrics is not None else None
        return state.name, elapsed, probability

    complete = incomplete = cut_incomplete = cut_complete = 0
    street_complete = street_incomplete = 0
    hesitation_incomplete = 0
    latencies: list[float] = []
    for row in clean:
        path = _resolve_audio(row)
        if path is None:
            continue
        pcm, rate = _read_wav(path)
        state, elapsed, _prob = await decide(pcm, rate, 1.0)
        latencies.append(elapsed)
        if state == EndOfTurnState.COMPLETE.name:
            complete += 1
        else:
            incomplete += 1
        cut_state, _, _ = await decide(pcm, rate, 0.45)
        if cut_state == EndOfTurnState.INCOMPLETE.name:
            cut_incomplete += 1
        else:
            cut_complete += 1
        hes_state, _, _ = await decide(pcm, rate, 0.30)
        if hes_state == EndOfTurnState.INCOMPLETE.name:
            hesitation_incomplete += 1
    for row in street:
        path = _resolve_audio(row)
        if path is None:
            continue
        pcm, rate = _read_wav(path)
        state, elapsed, _prob = await decide(pcm, rate, 1.0)
        latencies.append(elapsed)
        if state == EndOfTurnState.COMPLETE.name:
            street_complete += 1
        else:
            street_incomplete += 1
    await analyzer.cleanup()
    return {
        "available": True,
        "corpus": True,
        "clean_full_complete": complete,
        "clean_full_incomplete": incomplete,
        "clean_samples": complete + incomplete,
        "cut45_incomplete": cut_incomplete,
        "cut45_complete": cut_complete,
        "hesitation_cut30_incomplete": hesitation_incomplete,
        "street_full_complete": street_complete,
        "street_full_incomplete": street_incomplete,
        "street_samples": street_complete + street_incomplete,
        "eot_decision_p50_ms": _p50(latencies),
        "full_twilio_pipeline": UNMEASURED_TWILIO,
        "note": "Kokoro+mulaw corpus, not human telephone. cut45/cut30 proxy unfinished speech.",
    }


async def measure_pipecat_cleanup_and_tools() -> dict[str, Any]:
    settings = Settings(voice_eager_load=False, twilio_auth_token=SECRET)
    node = TrackingNode()
    runtime = VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=None,
        stt=None,
        tts=None,
        vad_model=None,
        engines_ready=True,
    )
    ws = FakeWs()
    await run_pipecat_session(
        ws,
        runtime,
        {
            "event": "start",
            "start": {
                "streamSid": "MZ1",
                "callSid": "CA1",
                "customParameters": {"instanceId": "restaurant-42"},
            },
        },
    )
    plan = await build_call_plan(
        {
            "event": "start",
            "start": {
                "streamSid": "MZ2",
                "callSid": "CA2",
                "customParameters": {
                    "instanceId": "restaurant-42",
                    "streamToken": create_stream_token("restaurant-42", SECRET),
                    "callerNumber": "+33612345678",
                },
            },
        },
        settings,
        node,
    )
    await _report_hangup(runtime, plan)
    await _report_hangup(runtime, plan)
    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self) -> None:
            self.fns: dict = {}

        def register_function(self, name, fn):
            self.fns[name] = fn

    llm = FakeLLM()
    await _register_fastify_tools(llm, runtime, plan.instance_id, plan.caller_number)

    async def capture(result):
        captured["result"] = result

    await llm.fns["create_appointment"](
        {
            "arguments": {
                "name": "Alice",
                "telephone": "0000000000",
                "date": "2026-09-20",
                "time": "21:00",
                "type": "Réservation de table",
                "nombrePersonnes": 6,
            },
            "result_callback": capture,
        }
    )
    update = await runtime.dispatcher.dispatch("update_appointment", {}, "restaurant-42")
    cancel = await runtime.dispatcher.dispatch("cancel_appointment", {}, "restaurant-42")
    return {
        "pipecat_installed": is_pipecat_available(),
        "token_reject_closes_ws": ws.closed == 1008,
        "hangups": len(node.hangups),
        "forced_caller": node.appointments[-1]["telephone"] == "0612345678" if node.appointments else False,
        "create_ok": bool(captured.get("result", {}).get("success")),
        "update_appointment": update,
        "cancel_appointment": cancel,
        "live_pipeline_twilio": UNMEASURED_TWILIO,
        "barge_in": UNMEASURED_TWILIO,
        "false_interruptions": UNMEASURED_TWILIO,
        "asyncio_tasks_after_hangup": UNMEASURED_TWILIO,
    }


def code_complexity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    native_files = [
        root / "session" / "call.py",
        root / "turn" / "detector.py",
        root / "audio" / "buffer.py",
        root / "twilio" / "protocol.py",
    ]
    shared = [root / "session" / "bootstrap.py"]
    pipecat_files = [root / "pipeline" / "pipecat_runner.py"]

    def lines(paths: list[Path]) -> int:
        total = 0
        for path in paths:
            if path.exists():
                total += len(path.read_text(encoding="utf-8").splitlines())
        return total

    version = None
    try:
        import pipecat

        version = getattr(pipecat, "__version__", "installed")
    except Exception:
        version = None
    return {
        "native_realtime_loc": lines(native_files),
        "pipecat_glue_loc": lines(pipecat_files),
        "shared_bootstrap_loc": lines(shared),
        "pipecat_version": version,
        "native_notes": (
            "CallSession owns VAD, heuristic EOT, 800ms hold, barge-in, Twilio clear, "
            "turn/incomplete tasks, hangup idempotent close."
        ),
        "pipecat_notes": (
            "Glue builds Twilio transport, WhisperSTT, OpenAILLM, KokoroTTS, Smart Turn v3, "
            "Fastify tools, greeting, hangup. allow_interruptions=True. "
            "_report_hangup is not idempotent. Live barge-in/EOT on Twilio not measured here."
        ),
    }


async def run_orchestrator_ab() -> dict[str, Any]:
    native_turn = await measure_native_turn()
    native_barge = await measure_native_barge_in()
    native_false_int = await measure_native_false_interrupt()
    native_clean = await measure_native_cleanup()
    native_tools = await measure_native_tools()
    smart = await measure_smart_turn()
    pipecat = await measure_pipecat_cleanup_and_tools()
    return {
        "hardware": "cpu",
        "stt_llm_tts": "fakes for native control plane; Smart Turn uses corpus wavs",
        "end_to_end_latency": "not used as orchestrator criterion (Whisper small CPU dominates)",
        "default_unchanged": "VOICE_ORCHESTRATOR=native",
        "native": {
            "turn": native_turn,
            "barge_in": native_barge,
            "false_interruptions": native_false_int,
            "cleanup": native_clean,
            "tools": native_tools,
            "eot": "heuristic + 800ms hold",
        },
        "pipecat": {
            "smart_turn": smart,
            "cleanup_tools": pipecat,
            "eot": "LocalSmartTurnAnalyzerV3",
            "full_twilio_pipeline": UNMEASURED_TWILIO,
        },
        "complexity": code_complexity(),
        "unmeasured": [
            "Native/Pipecat speech_end -> first_audio on real STT/LLM/TTS",
            "Pipecat live barge-in / Twilio clear / leftover audio",
            "Pipecat live false interruptions (souffle, resto, echo)",
            "Pipecat live pipeline close (WS, tasks, queues)",
            "Human telephone audio for Smart Turn",
            "update_appointment / cancel_appointment tools (absent both sides)",
        ],
    }
