import httpx
import pytest

from app.config import Settings
from app.llm import LLMResult
from app.llm.fallback import FallbackLLM
from app.llm.vllm import VLLMProvider


@pytest.mark.asyncio
async def test_fallback_gpt_when_vllm_fails(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("http://vllm.test"):
            return httpx.Response(503, json={"error": "vllm down"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ok gpt", "tool_calls": []}}
                ]
            },
        )

    settings = Settings(
        vllm_base_url="http://vllm.test/v1",
        vllm_model="local-model",
        openai_base_url="http://openai.test/v1",
        openai_model="gpt-4o-mini",
        openai_api_key="test",
    )
    transport = httpx.MockTransport(handler)
    primary = VLLMProvider(
        settings,
        client=httpx.AsyncClient(transport=transport, base_url="http://vllm.test/v1"),
    )
    fallback = VLLMProvider(
        settings.model_copy(
            update={"vllm_base_url": "http://openai.test/v1", "vllm_model": "gpt-4o-mini", "vllm_api_key": "test"}
        ),
        client=httpx.AsyncClient(transport=transport, base_url="http://openai.test/v1"),
    )
    llm = FallbackLLM(primary, fallback)
    caplog.set_level("WARNING")
    result = await llm.complete([{"role": "user", "content": "hi"}], [])
    assert isinstance(result, LLMResult)
    assert result.content == "ok gpt"
    assert llm.active_provider == "openai"
    assert any("FALLBACK GPT" in record.getMessage() for record in caplog.records)
    await llm.aclose()


@pytest.mark.asyncio
async def test_build_llm_without_openai_key_stays_on_vllm(caplog):
    from app.llm.fallback import build_llm
    from app.llm.vllm import VLLMProvider

    settings = Settings(
        llm_fallback_gpt=True,
        openai_api_key="",
        vllm_base_url="http://vllm.test/v1",
    )
    caplog.set_level("WARNING")
    llm = build_llm(settings)
    assert isinstance(llm, VLLMProvider)
    assert any("FALLBACK GPT" in record.getMessage() for record in caplog.records)
    await llm.aclose()


@pytest.mark.asyncio
async def test_startup_marks_engines_unready_when_vllm_down_without_gpt():
    from app.main import _log_vllm_model
    from app.node_client import NodeApiClient
    from app.runtime import VoiceRuntime
    from app.tools.dispatcher import ToolDispatcher

    class BoomLLM:
        async def list_models(self):
            raise RuntimeError("vllm down")

    settings = Settings(node_base_url="http://node.test")
    node = NodeApiClient(settings)
    runtime = VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=BoomLLM(),
        stt=None,
        tts=None,
        vad_model=None,
        engines_ready=True,
    )
    await _log_vllm_model(runtime, settings)
    assert runtime.engines_ready is False
    assert "vLLM" in (runtime.engine_error or "")
    await node.aclose()
