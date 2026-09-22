from __future__ import annotations

import json
import math
import re
from typing import Any

OFFICIAL_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

RESTAURANT_PRIMER_USER = (
    "Tu es l'assistante telephonique d'un restaurant francais. "
    "Tu parles uniquement francais, phrases courtes, une question a la fois, "
    "sans markdown, sans emoji, sans JSON."
)

RESTAURANT_PRIMER_ASSISTANT = "D'accord, je prends les appels du restaurant."

TOOL_PROMPT = (
    "Outils disponibles (ne les invente pas) :\n"
    "- check_availability(date: YYYY-MM-DD)\n"
    "- create_appointment(name, telephone, date, time, type, nombrePersonnes)\n"
    "Si un outil est necessaire, reponds uniquement avec "
    '<tool_call>{"name":"...","arguments":{...}}</tool_call> '
    "sans texte autour. Sinon reponds au client."
)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
BARE_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(check_availability|create_appointment|cancel_appointment)"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)
TOOL_LEAK_RE = re.compile(
    r"(tool_call|check_availability|create_appointment|cancel_appointment|\"arguments\")",
    re.IGNORECASE,
)
ENGLISH_ONLY_MARKERS = (
    "sure, i",
    "of course",
    "absolutely",
    "how can i help",
    "thank you for calling",
    "i would be happy",
)

FRENCH_CASES = (
    {
        "id": "reserve_quatre_20h",
        "text": (
            "Bonjour, je voudrais reserver une table pour quatre personnes "
            "ce soir a vingt heures."
        ),
        "expect_any": ("quatre", "4", "vingt", "20", "table", "reserv"),
        "expect_number": ("quatre", "4"),
        "expect_time": ("vingt", "20"),
    },
    {
        "id": "correction_six",
        "text": "Attendez, finalement nous serons six.",
        "expect_any": ("six", "6"),
        "expect_number": ("six", "6"),
        "expect_time": (),
    },
    {
        "id": "creneau_2030",
        "text": "Vous avez quelque chose vers vingt heures trente ?",
        "expect_any": ("vingt", "30", "trente", "heure"),
        "expect_number": ("trente", "30", "vingt"),
        "expect_time": ("vingt", "30", "trente", "20"),
    },
    {
        "id": "nom_nguyen",
        "text": "C'est au nom de Nguyen.",
        "expect_any": ("nguyen", "nom"),
        "expect_number": (),
        "expect_time": (),
    },
    {
        "id": "annuler_soir",
        "text": "Je voudrais annuler ma reservation de ce soir.",
        "expect_any": ("annul", "reserv", "soir"),
        "expect_number": (),
        "expect_time": (),
    },
)
FRENCH_MARKERS = (
    "bonjour",
    "bonsoir",
    "table",
    "reservation",
    "réserver",
    "reserver",
    "personnes",
    "heure",
    "disponib",
    "restaurant",
    "merci",
    "oui",
    "non",
    "combien",
    "nom",
)


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def build_conversation(
    *,
    user_text: str | None = None,
    audio_path: str | None = None,
    include_tools: bool = False,
    history: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": text_content(OFFICIAL_AUDIO_SYSTEM_PROMPT)},
        {"role": "user", "content": text_content(RESTAURANT_PRIMER_USER)},
        {"role": "assistant", "content": text_content(RESTAURANT_PRIMER_ASSISTANT)},
        *history,
    ]
    user_parts: list[dict[str, str]] = []
    if include_tools:
        user_parts.extend(text_content(TOOL_PROMPT))
    if audio_path:
        user_parts.append({"type": "audio", "audio": audio_path})
    if user_text:
        user_parts.extend(text_content(user_text))
    if not user_parts:
        raise ValueError("user_text or audio_path is required")
    messages.append({"role": "user", "content": user_parts})
    return messages


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []
    found: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        parsed = _load_tool_payload(match.group(1))
        if parsed is not None:
            found.append(parsed)
    if found:
        return found
    for match in BARE_JSON_RE.finditer(text):
        parsed = _load_tool_payload(match.group(0))
        if parsed is not None:
            found.append(parsed)
    return found


def _load_tool_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return {"name": name.strip(), "arguments": arguments}


def looks_like_french(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    return any(marker in lowered for marker in FRENCH_MARKERS)


def reply_quality_flags(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    return {
        "empty": cleaned == "",
        "looks_french": looks_like_french(cleaned),
        "has_tool_call": bool(parse_tool_calls(cleaned)),
        "char_count": len(cleaned),
        "too_long_for_phone": len(cleaned) > 280,
        "leaks_tool_json": leaks_tool_json(cleaned),
        "english_mix": has_english_mix(cleaned),
    }


def leaks_tool_json(text: str) -> bool:
    cleaned = text or ""
    if parse_tool_calls(cleaned):
        return True
    return bool(TOOL_LEAK_RE.search(cleaned)) or "{" in cleaned


def has_english_mix(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in ENGLISH_ONLY_MARKERS)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(needle.lower() in lowered for needle in needles) if needles else True


def score_french_case(case: dict[str, Any], thinker_text: str, spoken_text: str) -> dict[str, Any]:
    combined = f"{thinker_text} {spoken_text}"
    number_ok = _contains_any(combined, tuple(case.get("expect_number") or ()))
    time_ok = _contains_any(combined, tuple(case.get("expect_time") or ()))
    intent_ok = _contains_any(combined, tuple(case.get("expect_any") or ()))
    spoken_ok = looks_like_french(spoken_text) and not leaks_tool_json(spoken_text)
    name_ok = True
    if case["id"] == "nom_nguyen":
        name_ok = "nguyen" in combined.lower()
    return {
        "id": case["id"],
        "intent_ok": intent_ok,
        "number_ok": number_ok,
        "time_ok": time_ok,
        "name_ok": name_ok,
        "spoken_french": spoken_ok,
        "english_mix": has_english_mix(spoken_text) or has_english_mix(thinker_text),
        "leaks_tool_json": leaks_tool_json(spoken_text),
        "acceptable": intent_ok and number_ok and time_ok and name_ok and spoken_ok,
    }


def fake_fastify(tool_call: dict[str, Any]) -> dict[str, Any]:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    if name == "check_availability":
        spoken = "Oui, il reste de la place ce soir vers vingt heures."
        return {
            "ok": True,
            "name": name,
            "arguments": arguments,
            "spoken": spoken,
            "payload": {"windows": ["20:00-21:30"], "remainingCoversSoir": 8},
        }
    if name == "create_appointment":
        spoken = "C'est note, table pour ce soir."
        return {
            "ok": True,
            "name": name,
            "arguments": arguments,
            "spoken": spoken,
            "payload": {"status": "created"},
        }
    if name == "cancel_appointment":
        spoken = "D'accord, j'annule la reservation de ce soir."
        return {
            "ok": True,
            "name": name,
            "arguments": arguments,
            "spoken": spoken,
            "payload": {"status": "cancelled"},
        }
    return {
        "ok": False,
        "name": name,
        "arguments": arguments,
        "spoken": "Je n'ai pas pu faire cette action.",
        "payload": {"status": "unknown_tool"},
    }


def talker_user_text(spoken_result: str) -> str:
    return (
        "Dis ceci au client, mot pour mot, en francais, sans ajouter de JSON "
        f"ni de nom d'outil : {spoken_result}"
    )


def decide_integration(
    *,
    oom: bool,
    french_quality: str,
    tool_call_success: bool,
    output_audio_ok: bool,
    thinker_p50_ms: float | None,
    total_p50_ms: float | None,
) -> dict[str, Any]:
    quality_go = (
        (not oom)
        and french_quality == "acceptable"
        and tool_call_success
        and output_audio_ok
    )
    # generate() is not realtime. Streaming work is worth it if thinker
    # returns usable text well below a full telephone turn.
    latency_justifies_streaming = (
        thinker_p50_ms is not None and thinker_p50_ms < 4000
    )
    notes = []
    if oom:
        notes.append("OOM or load failure.")
    if french_quality != "acceptable":
        notes.append("French quality below the heuristic bar.")
    if not tool_call_success:
        notes.append("Thinker tool loop failed or Talker leaked JSON.")
    if not output_audio_ok:
        notes.append("Output wav missing, silent, or unusable.")
    if thinker_p50_ms is None:
        notes.append("Thinker latency not measured.")
    elif not latency_justifies_streaming:
        notes.append(
            f"Thinker p50 {thinker_p50_ms} ms is too slow to justify a streaming prototype."
        )
    if total_p50_ms is not None:
        notes.append(
            f"Full generate() p50 is {total_p50_ms} ms and is not a Twilio SLA."
        )
    return {
        "integration_allowed": quality_go and latency_justifies_streaming,
        "quality_go": quality_go,
        "latency_justifies_streaming": latency_justifies_streaming,
        "notes": notes,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def compatibility_audit() -> dict[str, Any]:
    """Facts from official Qwen cards. Measured L4 numbers stay null until CUDA runs."""
    return {
        "model": "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
        "target_gpu": "NVIDIA L4 24GB",
        "sources": [
            "https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
            "https://github.com/QwenLM/Qwen2.5-Omni",
            "https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_omni",
        ],
        "vram": {
            "official_int4_15s_video_gb": 11.64,
            "official_int4_30s_video_gb": 17.43,
            "official_int4_60s_video_gb": 29.51,
            "bf16_7b_15s_video_gb": 31.11,
            "l4_fits_int4_15s_video": True,
            "l4_fits_int4_30s_video": True,
            "l4_fits_int4_60s_video": False,
            "l4_fits_bf16_7b": False,
            "audio_only_note": (
                "Official numbers are video. Audio-only turns must stay short. "
                "Keep thinker.visual on CPU. Do not keep 60s of audio context."
            ),
            "cpu_offload_note": (
                "Official INT4 demo offloads visual and audio_tower to CPU. "
                "That saves VRAM and adds latency. Phone S2S should keep audio_tower on CUDA."
            ),
            "measured_peak_gb": None,
        },
        "streaming": {
            "architecture": "Thinker-Talker plus sliding-window DiT",
            "transformers_generate": (
                "Returns the full 24 kHz waveform when generate() finishes. "
                "Not a Twilio 20 ms frame iterator."
            ),
            "official_int4_token2wav": (
                "Internal streaming exists to avoid pre-allocating VRAM. "
                "The Python API still waits for the complete audio."
            ),
            "vllm_openai_current_stack": (
                "vllm/vllm-openai:v0.10.2 serves thinker text only. No talker, no S2S."
            ),
            "vllm_omni": (
                "Separate project. Text can stream. Audio is returned at the end of the stage."
            ),
            "dashscope_cloud": "True audio streaming exists in the cloud API. That is not the L4 target.",
            "measured_s2s_total_ms": None,
        },
        "function_calling": {
            "local_generate": "No first-class tools argument on official generate().",
            "thinker_path": (
                "Thinker is Qwen2.5. A prompt-level tool loop with return_audio=False "
                "is the only local path that can call Fastify."
            ),
            "s2s_risk": (
                "If tools stay on the speech path, Talker may read tool JSON aloud."
            ),
            "cloud_dashscope": "Function calling is documented on the hosted Omni API only.",
            "measured": None,
        },
        "french": {
            "asr_common_voice_15_fr_wer_official": 7.5,
            "tts_official_eval": "Seed-TTS zh/en, not French telephone.",
            "voices": ["Chelsie", "Ethan"],
            "system_prompt_constraint": (
                "Audio output requires the official Qwen system prompt. "
                "Restaurant style must be primed as a user/assistant turn."
            ),
            "measured": None,
        },
        "cannot_cohabit_with": [
            "faster-whisper",
            "vLLM Qwen2.5-3B-Instruct",
            "Kokoro",
            "Smart Turn as a second orchestrator",
        ],
    }
