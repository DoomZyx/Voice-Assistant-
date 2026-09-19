from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.session.bootstrap import apply_forced_caller_number, build_call_plan, sanitize_identifier


logger = logging.getLogger("mysmartfood-voice.pipecat")

# Recommandation Pipecat Smart Turn : VAD court, analyse du tour 16 kHz complet.
PIPECAT_TWILIO_HZ = 8000
PIPECAT_AUDIO_IN_HZ = 16000
PIPECAT_VAD_STOP_SECS = 0.2


def pipecat_turn_settings() -> dict[str, float | int]:
    return {
        "twilio_hz": PIPECAT_TWILIO_HZ,
        "audio_in_hz": PIPECAT_AUDIO_IN_HZ,
        "vad_stop_secs": PIPECAT_VAD_STOP_SECS,
    }

_IMPORT_ERROR: str | None
try:
    import pipecat  # noqa: F401

    _PIPECAT_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - absence du paquet
    _PIPECAT_AVAILABLE = False
    _IMPORT_ERROR = str(exc)


def is_pipecat_available() -> bool:
    return _PIPECAT_AVAILABLE


def pipecat_import_error() -> str | None:
    return _IMPORT_ERROR


def should_use_pipecat(settings: Settings, available: bool | None = None) -> bool:
    requested = (settings.voice_orchestrator or "native").strip().lower() == "pipecat"
    installed = _PIPECAT_AVAILABLE if available is None else available
    return requested and installed


def extract_twilio_ids(start_event: dict[str, Any] | None) -> tuple[str, str, str | None]:
    start = (start_event or {}).get("start") or {}
    custom = start.get("customParameters") or {}
    if not isinstance(custom, dict):
        custom = {}
    instance_id = sanitize_identifier(custom.get("instanceId"))
    return str(start.get("streamSid") or ""), str(start.get("callSid") or ""), instance_id


async def run_pipecat_session(websocket: Any, runtime: Any, start_event: dict[str, Any] | None = None) -> None:
    """Orchestration Pipecat. Meme streamToken, contexte et tools que CallSession."""
    plan = await build_call_plan(
        start_event,
        runtime.settings,
        runtime.node,
        getattr(runtime, "telemetry", None),
    )
    if plan.rejected:
        logger.warning("step=pipecat.start.rejected reason=%s", plan.reject_reason)
        close = getattr(websocket, "close", None)
        if callable(close):
            await close(code=1008)
        return
    if not _PIPECAT_AVAILABLE:
        raise RuntimeError(
            "VOICE_ORCHESTRATOR=pipecat mais pipecat-ai est absent. "
            "pip install 'mysmartfood-voice[pipecat]' ou VOICE_ORCHESTRATOR=native."
        )
    await _run_installed_pipecat(websocket, runtime, plan)


async def _run_installed_pipecat(websocket: Any, runtime: Any, plan: Any) -> None:
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    settings: Settings = runtime.settings
    stream_sid = plan.stream_sid
    call_sid = plan.call_sid
    instance_id = plan.instance_id
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        params=TwilioFrameSerializer.InputParams(
            auto_hang_up=False,
            twilio_sample_rate=PIPECAT_TWILIO_HZ,
            sample_rate=PIPECAT_AUDIO_IN_HZ,
        ),
    )
    def _silero_vad():
        return SileroVADAnalyzer(
            sample_rate=PIPECAT_AUDIO_IN_HZ,
            params=VADParams(stop_secs=PIPECAT_VAD_STOP_SECS),
        )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            vad_analyzer=_silero_vad(),
        ),
    )
    stt = WhisperSTTService(
        model=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        settings=WhisperSTTService.Settings(language=Language.FR),
    )
    llm = OpenAILLMService(
        api_key=settings.vllm_api_key or "not-needed",
        base_url=settings.vllm_base_url,
        model=settings.vllm_model,
    )
    if plan.tools_enabled:
        await _register_fastify_tools(llm, runtime, instance_id, plan.caller_number)
    from pipecat.services.kokoro.tts import KokoroTTSService

    kokoro = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=settings.kokoro_voice,
            language=Language.FR,
        )
    )
    seed = list(plan.seed_messages)
    try:
        context = LLMContext(messages=seed, tools=plan.llm_tools)
    except TypeError:
        context = LLMContext(messages=seed)
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=_silero_vad(),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(sample_rate=PIPECAT_AUDIO_IN_HZ)
                    )
                ]
            ),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            kokoro,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=PIPECAT_AUDIO_IN_HZ,
            audio_out_sample_rate=PIPECAT_TWILIO_HZ,
            enable_metrics=True,
            allow_interruptions=True,
        ),
    )
    runner = PipelineRunner()

    @transport.event_handler("on_client_connected")
    async def _greet(_transport: Any, _client: Any) -> None:
        if not plan.greeting:
            return
        try:
            from pipecat.frames.frames import TTSSpeakFrame

            await task.queue_frames([TTSSpeakFrame(plan.greeting)])
        except Exception:
            logger.warning("step=pipecat.greeting.skip")

    logger.info(
        "step=pipecat.start stream=%s call=%s instance=%s tools=%s stt=%s llm=%s tts=kokoro",
        stream_sid,
        call_sid,
        instance_id,
        plan.tools_enabled,
        settings.whisper_model_size,
        settings.vllm_model,
    )
    try:
        await runner.run(task)
    finally:
        await _report_hangup(runtime, plan)


async def _register_fastify_tools(
    llm: Any,
    runtime: Any,
    instance_id: str | None,
    caller_number: str | None = None,
) -> None:
    """LLM -> fonction Pipecat -> ToolDispatcher HTTP -> Fastify. Jamais SQL."""
    register = getattr(llm, "register_function", None)
    dispatcher = getattr(runtime, "dispatcher", None)
    if not callable(register) or dispatcher is None:
        logger.warning("step=pipecat.tools.skip reason=register_or_dispatcher_missing")
        return

    async def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await dispatcher.dispatch(name, arguments, instance_id)

    async def check_availability(params: Any) -> None:
        arguments = _tool_arguments(params)
        result = await _dispatch("check_availability", arguments)
        await _tool_result(params, result)

    async def create_appointment(params: Any) -> None:
        arguments = apply_forced_caller_number(_tool_arguments(params), caller_number)
        result = await _dispatch("create_appointment", arguments)
        await _tool_result(params, result)

    register("check_availability", check_availability)
    register("create_appointment", create_appointment)
    logger.info("step=pipecat.tools.registered instance=%s", instance_id)


def _tool_arguments(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        raw = params.get("arguments") or params.get("args") or {}
        return raw if isinstance(raw, dict) else {}
    raw = getattr(params, "arguments", None) or getattr(params, "args", None) or {}
    return raw if isinstance(raw, dict) else {}


async def _tool_result(params: Any, result: dict[str, Any]) -> None:
    callback = getattr(params, "result_callback", None)
    if callback is None and isinstance(params, dict):
        callback = params.get("result_callback")
    if callable(callback):
        await callback(result)


def _hangup_key(plan: Any) -> tuple[str | None, str, str]:
    return (plan.instance_id, str(plan.stream_sid or ""), str(plan.call_sid or ""))


async def _report_hangup(runtime: Any, plan: Any) -> None:
    if not plan.instance_id:
        return
    report = getattr(getattr(runtime, "node", None), "report_call_ended", None)
    if not callable(report):
        return
    reported = getattr(runtime, "_pipecat_hangups", None)
    if reported is None:
        reported = set()
        runtime._pipecat_hangups = reported
    key = _hangup_key(plan)
    if key in reported:
        logger.info("step=pipecat.hangup.idempotent instance=%s", plan.instance_id)
        return
    try:
        await report(
            {
                "instanceId": plan.instance_id,
                "streamSid": plan.stream_sid,
                "callSid": plan.call_sid,
                "turns": [],
                "liveCreate": None,
            },
            tenant_id=plan.instance_id,
        )
        reported.add(key)
    except Exception as exc:
        logger.warning("step=pipecat.hangup.failed error=%s", str(exc)[:200])
