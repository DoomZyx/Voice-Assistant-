import pytest

from app.config import Settings
from app.preflight import check_fastapi, check_vllm


@pytest.mark.asyncio
async def test_preflight_fastapi_starts_without_engines():
    settings = Settings(voice_eager_load=False, node_base_url="http://node.test")
    result = await check_fastapi(settings)
    assert result.ok is True
    assert result.name == "fastapi"


@pytest.mark.asyncio
async def test_preflight_vllm_reports_model_mismatch(monkeypatch):
    class FakeLLM:
        async def list_models(self):
            return ["meta-llama/Llama-3.1-8B-Instruct"]

        async def complete(self, messages, tools):
            raise AssertionError("complete ne doit pas etre appele si l'id est faux")

        async def aclose(self):
            return None

    monkeypatch.setattr("app.llm.vllm.VLLMProvider", lambda settings: FakeLLM())
    settings = Settings(vllm_model="llama-3.1-8b-instruct")
    result = await check_vllm(settings)
    assert result.ok is False
    assert "meta-llama/Llama-3.1-8B-Instruct" in result.detail
