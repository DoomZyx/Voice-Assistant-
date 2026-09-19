import pytest

from app.config import Settings
from app.conversation.corrections import latest_party_size
from app.llm import LLMResult, ToolCall
from app.runtime import VoiceRuntime
from app.session.call import CallSession
from app.tools.dispatcher import ToolDispatcher
from tests.fakes import FakeSTT, FakeTTS, FakeVadModel, FakeWebSocket, FakeNode


class ScriptedReservationLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_user: list[str] = []

    async def complete(self, messages, tools):
        self.calls += 1
        user_texts = [item["content"] for item in messages if item.get("role") == "user"]
        self.seen_user = [text for text in user_texts if isinstance(text, str)]
        last = self.seen_user[-1] if self.seen_user else ""
        last_user = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=-1,
        )
        if any(item.get("role") == "tool" for item in messages[last_user + 1 :]):
            return LLMResult(content="J'ai 20h15 ou 20h30.")
        if "vingt et une" in last or "21" in last:
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id="avail_21",
                        name="check_availability",
                        arguments={"date": "2026-09-20"},
                    )
                ]
            )
        if any("six" in text.lower() for text in self.seen_user):
            return LLMResult(content="Oui. A quel nom ?")
        if "quatre" in last and "demain" in last:
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id="avail_1",
                        name="check_availability",
                        arguments={"date": "2026-09-20"},
                    )
                ]
            )
        return LLMResult(content="Pour combien de personnes ?")


def _runtime(llm) -> VoiceRuntime:
    settings = Settings(voice_eager_load=False)
    node = FakeNode()
    return VoiceRuntime(
        settings=settings,
        node=node,
        dispatcher=ToolDispatcher(node),
        llm=llm,
        stt=FakeSTT(),
        tts=FakeTTS(),
        vad_model=FakeVadModel(),
        engines_ready=True,
    )


def test_latest_party_size_uses_the_correction():
    assert latest_party_size(["Nous serons quatre.", "Ah non, six."]) == 6
    assert latest_party_size(["Finalement on sera six."]) == 6


@pytest.mark.asyncio
async def test_eval_reservation_calls_check_availability_not_create():
    llm = ScriptedReservationLLM()
    runtime = _runtime(llm)
    session = CallSession(FakeWebSocket(), runtime)
    await session._think("Je voudrais reserver pour quatre demain a 20 heures.")
    assert runtime.node.availability_calls == ["2026-09-20"]
    assert runtime.node.appointments == []
    await session.close()


@pytest.mark.asyncio
async def test_eval_availability_does_not_invent_confirmation():
    llm = ScriptedReservationLLM()
    runtime = _runtime(llm)
    session = CallSession(FakeWebSocket(), runtime)
    reply = await session._think("Je voudrais reserver pour quatre demain a 20 heures.")
    assert "confirmee" not in reply.lower()
    assert "20h15" in reply or "20h30" in reply
    await session.close()


@pytest.mark.asyncio
async def test_eval_correction_keeps_six_in_history():
    llm = ScriptedReservationLLM()
    session = CallSession(FakeWebSocket(), _runtime(llm))
    await session._think("Nous serons quatre.")
    reply = await session._think("Ah non, six.")
    assert latest_party_size(llm.seen_user) == 6
    assert "nom" in reply.lower()
    await session.close()


@pytest.mark.asyncio
async def test_eval_interruption_requests_new_availability():
    llm = ScriptedReservationLLM()
    runtime = _runtime(llm)
    session = CallSession(FakeWebSocket(), runtime)
    await session._think("Je voudrais reserver pour quatre demain a 20 heures.")
    runtime.node.availability_calls.clear()
    await session._think("Plutot vingt et une heures.")
    assert runtime.node.availability_calls == ["2026-09-20"]
    await session.close()
