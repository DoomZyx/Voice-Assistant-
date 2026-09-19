import httpx
import pytest

from livekit_agent.config import LiveKitSettings
from livekit_agent.llm.config import model_id_error
from livekit_agent.preflight import LLMModelMismatch, assert_llm_model


def test_model_id_must_match_exactly():
    served = ["mistralai/Mistral-7B-Instruct-v0.3"]
    assert model_id_error("mistralai/Mistral-7B-Instruct-v0.3", served) is None
    error = model_id_error("mistral-7b", served)
    assert error is not None
    assert "mistralai/Mistral-7B-Instruct-v0.3" in error


def test_model_id_empty_and_no_served():
    assert model_id_error("", ["x"]) is not None
    assert model_id_error("x", []) is not None


@pytest.mark.asyncio
async def test_preflight_raises_and_logs_mismatch(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "served-model"}]})
        return httpx.Response(404)

    settings = LiveKitSettings(
        llm_base_url="http://vllm.test/v1",
        llm_model="wrong-model",
        llm_api_key="k",
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.test/v1",
    )
    with caplog.at_level("ERROR"):
        with pytest.raises(LLMModelMismatch) as exc:
            await assert_llm_model(settings, client=http)
    await http.aclose()
    assert "wrong-model" in str(exc.value)
    assert any("step=vllm.model_mismatch" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_preflight_accepts_exact_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "served-model"}]})

    settings = LiveKitSettings(llm_base_url="http://vllm.test/v1", llm_model="served-model")
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.test/v1",
    )
    served = await assert_llm_model(settings, client=http)
    await http.aclose()
    assert served == ["served-model"]
