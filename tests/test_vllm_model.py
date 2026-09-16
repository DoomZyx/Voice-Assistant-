import httpx
import pytest

from app.llm.vllm import VLLMProvider, model_id_error


def test_model_id_must_match_exactly():
    served = ["meta-llama/Llama-3.1-8B-Instruct"]
    assert model_id_error("meta-llama/Llama-3.1-8B-Instruct", served) is None
    error = model_id_error("llama-3.1-8b-instruct", served)
    assert error is not None
    assert "meta-llama/Llama-3.1-8B-Instruct" in error


def test_model_id_empty_and_no_served():
    assert model_id_error("", ["x"]) is not None
    assert model_id_error("x", []) is not None


@pytest.mark.asyncio
async def test_vllm_list_models_and_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]},
            )
        return httpx.Response(404, json={"error": "not found"})

    from app.config import Settings

    settings = Settings(vllm_model="llama-3.1-8b-instruct", vllm_base_url="http://vllm.test/v1")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm.test/v1")
    llm = VLLMProvider(settings, client=http)
    served = await llm.list_models()
    assert served == ["meta-llama/Llama-3.1-8B-Instruct"]
    assert model_id_error(settings.vllm_model, served) is not None
    await llm.aclose()


@pytest.mark.asyncio
async def test_hermes_xml_in_content_becomes_tool_call():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "restaurant-assistant",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<tool_call>{"name": "create_appointment", '
                                '"arguments": {"name": "Ada", "type": "Réservation de table"}}'
                                "</tool_call>"
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    from app.config import Settings

    settings = Settings(vllm_model="restaurant-assistant", vllm_base_url="http://vllm.test/v1")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm.test/v1")
    llm = VLLMProvider(settings, client=http)
    result = await llm.complete([{"role": "user", "content": "oui"}], [])
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "create_appointment"
    assert result.tool_calls[0].arguments["name"] == "Ada"
    await llm.aclose()
