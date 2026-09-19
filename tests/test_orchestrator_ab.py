from pathlib import Path

import pytest

from app.ab.orchestrator import (
    UNMEASURED_TWILIO,
    measure_native_cleanup,
    measure_native_tools,
    measure_native_turn,
    measure_pipecat_cleanup_and_tools,
    run_orchestrator_ab,
)
from app.config import Settings


@pytest.mark.asyncio
async def test_native_incomplete_phrase_is_held_not_answered_immediately():
    result = await measure_native_turn(repeats=2)
    assert result["immediate_false_eot"] == 0
    assert result["combined_before_hold"] == 2
    assert result["hold_ms"] == 800


@pytest.mark.asyncio
async def test_native_close_reports_hangup_once():
    result = await measure_native_cleanup(repeats=2)
    assert result["cleanup_failures"] == 0
    assert result["double_hangup"] == 0


@pytest.mark.asyncio
async def test_tools_force_caller_and_lack_update_cancel_on_native():
    result = await measure_native_tools()
    assert result["check_availability_ok"] is True
    assert result["create_appointment_ok"] is True
    assert result["forced_caller"] is True
    assert result["update_appointment"]["success"] is False
    assert result["cancel_appointment"]["success"] is False


@pytest.mark.asyncio
async def test_pipecat_reject_and_tools_match_native_contract():
    result = await measure_pipecat_cleanup_and_tools()
    assert result["token_reject_closes_ws"] is True
    assert result["forced_caller"] is True
    assert result["create_ok"] is True
    assert result["update_appointment"]["success"] is False
    assert result["cancel_appointment"]["success"] is False
    assert result["live_pipeline_twilio"] == UNMEASURED_TWILIO


@pytest.mark.asyncio
async def test_ab_payload_marks_unmeasured_and_keeps_native_default():
    payload = await run_orchestrator_ab()
    assert payload["default_unchanged"] == "VOICE_ORCHESTRATOR=native"
    assert payload["pipecat"]["full_twilio_pipeline"] == UNMEASURED_TWILIO
    assert "Native/Pipecat speech_end -> first_audio on real STT/LLM/TTS" in payload["unmeasured"]
    example = Path(__file__).resolve().parents[1] / ".env.example"
    assert "VOICE_ORCHESTRATOR=native" in example.read_text(encoding="utf-8")
    assert Settings(voice_eager_load=False).voice_orchestrator == "native"
