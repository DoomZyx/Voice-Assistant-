from __future__ import annotations

import asyncio
import logging
import sys

from livekit_agent.config import get_settings
from livekit_agent.phase4 import missing_runtime_settings, phase4_readiness, worker_exit_code
from livekit_agent.preflight import LLMModelMismatch, assert_llm_model
from livekit_agent.runtime import start_voice_session

logger = logging.getLogger("mysmartfood-voice.livekit")


async def restaurant_voice(ctx) -> None:
    await start_voice_session(ctx, get_settings())


def build_server(settings=None):
    from livekit.agents import AgentServer

    resolved = settings or get_settings()
    server = AgentServer()
    server.rtc_session(agent_name=resolved.agent_name)(restaurant_voice)
    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = get_settings()
    report = phase4_readiness(settings)
    if settings.voice_tools_enabled and report.blockers:
        logger.warning("step=phase4.skipped tools=true blockers=%s", ",".join(report.blockers))
    elif settings.voice_tools_enabled:
        logger.info("step=tools.profile profile=%s", (settings.voice_tools_profile or "").strip() or "-")
    if worker_exit_code(settings):
        logger.error(
            "step=worker.config_missing missing=%s",
            ",".join(missing_runtime_settings(settings)) or "-",
        )
        sys.exit(1)
    if not report.ready:
        logger.warning(
            "step=phase4.readiness ready=false missing=%s blockers=%s",
            ",".join(report.missing) or "-",
            ",".join(report.blockers) or "-",
        )
    try:
        asyncio.run(assert_llm_model(settings))
    except LLMModelMismatch as exc:
        logger.error("step=vllm.model_mismatch error=%s", exc)
        sys.exit(1)

    from livekit.agents import cli

    if len(sys.argv) == 1:
        sys.argv.append("start")
    cli.run_app(build_server(settings))


if __name__ == "__main__":
    main()
