"""Isolated Qwen2.5-Omni INT4 lab. Not wired to Twilio or Fastify.

  python -m prototypes.omni_s2s.proto_s2s --dry-run
  python -m prototypes.omni_s2s.proto_s2s --force --reps 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from prototypes.omni_s2s.helpers import (
    FRENCH_CASES,
    build_conversation,
    compatibility_audit,
    decide_integration,
    fake_fastify,
    parse_tool_calls,
    percentile,
    reply_quality_flags,
    score_french_case,
    talker_user_text,
)

MODEL_ID = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
OUTPUT_RATE = 24000
THINKER_MAX_NEW_TOKENS = 96
TALKER_MAX_NEW_TOKENS = 192
USED_MEMORY_REFUSE_MIB = 4096
S2S_USER_TEXT = (
    "Reponds au client, en francais, comme au telephone. "
    "Pas de JSON, pas de nom d'outil."
)
TOOL_AUDIO_HINT = (
    "Le client parle. Si un outil est necessaire, reponds uniquement avec "
    "un bloc tool_call. Sinon reponds au client."
)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _smi_query(fields: str) -> str | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    return line or None


def _smi_used_mib() -> int | None:
    raw = _smi_query("memory.used")
    if raw is None:
        return None
    try:
        return int(float(raw.split(",")[0].strip()))
    except ValueError:
        return None


def _read_cpu_times() -> tuple[int, int] | None:
    try:
        line = Path("/proc/stat").read_text().splitlines()[0]
    except OSError:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    values = [int(item) for item in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


class ResourceSampler:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu: list[float] = []
        self.cpu: list[float] = []
        self._prev_cpu = _read_cpu_times()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return {
            "gpu_util_avg": _mean(self.gpu),
            "gpu_util_max": max(self.gpu) if self.gpu else None,
            "cpu_util_avg": _mean(self.cpu),
            "cpu_util_max": max(self.cpu) if self.cpu else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            gpu_raw = _smi_query("utilization.gpu")
            if gpu_raw is not None:
                try:
                    self.gpu.append(float(gpu_raw.split(",")[0].strip()))
                except ValueError:
                    pass
            current = _read_cpu_times()
            if current and self._prev_cpu:
                idle_d = current[0] - self._prev_cpu[0]
                total_d = current[1] - self._prev_cpu[1]
                if total_d > 0:
                    self.cpu.append(max(0.0, min(100.0, (1.0 - idle_d / total_d) * 100.0)))
            self._prev_cpu = current
            self._stop.wait(self.interval_s)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch absent"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda unavailable"}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    reserved = torch.cuda.memory_reserved(index)
    allocated = torch.cuda.memory_allocated(index)
    peak = torch.cuda.max_memory_allocated(index)
    total = props.total_memory
    return {
        "available": True,
        "name": props.name,
        "total_gb": round(total / 1024**3, 2),
        "allocated_gb": round(allocated / 1024**3, 2),
        "reserved_gb": round(reserved / 1024**3, 2),
        "peak_allocated_gb": round(peak / 1024**3, 2),
        "free_gb": round((total - reserved) / 1024**3, 2),
        "smi_used_mib": _smi_used_mib(),
    }


def _device_map(mode: str) -> dict[str, str]:
    audio_device = "cpu" if mode == "official_low_vram" else "cuda"
    return {
        "thinker.model": "cuda",
        "thinker.lm_head": "cuda",
        "thinker.visual": "cpu",
        "thinker.audio_tower": audio_device,
        "talker": "cuda",
        "token2wav": "cuda",
    }


def _decode_text(processor: Any, token_ids: Any) -> str:
    decoded = processor.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = decoded[0] if decoded else ""
    marker = "assistant\n"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    return text.strip()


def _write_wav(path: Path, audio: Any) -> int:
    import soundfile as sf

    waveform = audio.reshape(-1).detach().cpu().float().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, samplerate=OUTPUT_RATE)
    return int(waveform.shape[0])


def _wav_duration_ms(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return round((info.frames / float(info.samplerate)) * 1000.0, 1)


def _prepare_inputs(processor: Any, conversation: list[dict[str, Any]], model: Any) -> Any:
    from qwen_omni_utils import process_mm_info

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return inputs.to("cuda").to(model.dtype)


def load_model(model_id: str, mode: str, attn: str) -> tuple[Any, Any]:
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=_device_map(mode),
            attn_implementation=attn,
        )
    except Exception as exc:
        raise RuntimeError(
            "Transformers failed to load the GPTQ checkpoint. "
            "Install prototypes/omni_s2s/requirements.txt."
        ) from exc
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return model, processor


def generate_turn(
    model: Any,
    processor: Any,
    conversation: list[dict[str, Any]],
    *,
    speaker: str,
    return_audio: bool,
    wav_path: Path | None,
) -> dict[str, Any]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    inputs = _prepare_inputs(processor, conversation, model)
    sampler.start()
    started = _now_ms()
    first_text_ms = None
    try:
        output = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=return_audio,
            speaker=speaker,
            thinker_max_new_tokens=THINKER_MAX_NEW_TOKENS,
            talker_max_new_tokens=TALKER_MAX_NEW_TOKENS,
        )
    finally:
        resources = sampler.stop()
    elapsed_ms = round(_now_ms() - started, 1)
    if return_audio:
        token_ids = output[0]
        audio = output[-1]
        samples = _write_wav(wav_path, audio) if wav_path is not None else 0
    else:
        token_ids = output
        samples = 0
    text = _decode_text(processor, token_ids)
    if text:
        first_text_ms = elapsed_ms
    snapshot = _cuda_snapshot()
    audio_ms = round(samples * 1000 / OUTPUT_RATE, 1) if samples else 0.0
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "first_usable_text_ms": first_text_ms,
        "audio_generation_start_ms": None,
        "audio_samples": samples,
        "audio_ms": audio_ms,
        "rtf": round(elapsed_ms / audio_ms, 3) if audio_ms else None,
        "peak_allocated_gb": snapshot.get("peak_allocated_gb"),
        "quality": reply_quality_flags(text),
        "tool_calls": parse_tool_calls(text),
        "wav": str(wav_path) if wav_path is not None and return_audio else None,
        "resources": resources,
    }


def synthesize_input(text: str, path: Path, model: Any, processor: Any, speaker: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    espeak = subprocess.run(
        ["espeak-ng", "-v", "fr", "-s", "140", "-w", str(path), text],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if espeak.returncode == 0 and path.exists():
        return {
            "path": str(path),
            "source": "espeak-ng",
            "duration_ms": _wav_duration_ms(path),
        }
    seed = generate_turn(
        model,
        processor,
        build_conversation(user_text=text),
        speaker=speaker,
        return_audio=True,
        wav_path=path,
    )
    return {
        "path": str(path),
        "source": "omni_loopback",
        "duration_ms": seed["audio_ms"],
        "seed_text": seed["text"],
    }


def run_measured(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a measured run")

    before = _cuda_snapshot()
    used_mib = _smi_used_mib()
    if used_mib is not None and used_mib > USED_MEMORY_REFUSE_MIB and not args.force:
        raise RuntimeError(
            f"nvidia-smi reports {used_mib} MiB in use. "
            "Stop the Whisper/vLLM/Kokoro stack, then rerun, or pass --force."
        )

    load_started = _now_ms()
    model, processor = load_model(args.model, args.mode, args.attn)
    load_ms = round(_now_ms() - load_started, 1)
    after_load = _cuda_snapshot()
    out_dir = Path(args.wav_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup = generate_turn(
        model,
        processor,
        build_conversation(user_text="Bonjour."),
        speaker=args.speaker,
        return_audio=True,
        wav_path=out_dir / "warmup.wav",
    )

    french_rows: list[dict[str, Any]] = []
    timed_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for case in FRENCH_CASES:
        source = synthesize_input(
            case["text"],
            out_dir / f"in_{case['id']}.wav",
            model,
            processor,
            args.speaker,
        )
        thinker = generate_turn(
            model,
            processor,
            build_conversation(
                audio_path=source["path"],
                user_text=S2S_USER_TEXT,
                history=tuple(history),
            ),
            speaker=args.speaker,
            return_audio=False,
            wav_path=None,
        )
        spoken = generate_turn(
            model,
            processor,
            build_conversation(
                audio_path=source["path"],
                user_text=S2S_USER_TEXT,
                history=tuple(history),
            ),
            speaker=args.speaker,
            return_audio=True,
            wav_path=out_dir / f"out_{case['id']}_{args.speaker.lower()}.wav",
        )
        score = score_french_case(case, thinker["text"], spoken["text"])
        row = {
            "id": case["id"],
            "prompt": case["text"],
            "input": source,
            "thinker": thinker,
            "spoken": spoken,
            "score": score,
            "metrics": {
                "audio_input_duration_ms": source["duration_ms"],
                "audio_input_to_thinker_ms": thinker["latency_ms"],
                "audio_input_to_first_usable_text_ms": thinker["first_usable_text_ms"],
                "audio_input_to_audio_generation_start_ms": None,
                "audio_input_to_completed_wav_ms": spoken["latency_ms"],
                "output_audio_duration_ms": spoken["audio_ms"],
                "rtf": spoken["rtf"],
                "thinker_latency_ms": thinker["latency_ms"],
                "talker_latency_ms": spoken["latency_ms"],
            },
        }
        french_rows.append(row)
        history = [
            *history,
            {"role": "user", "content": [{"type": "text", "text": case["text"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": spoken["text"]}]},
        ]

    timed_case = FRENCH_CASES[0]
    timed_source = french_rows[0]["input"]
    for index in range(args.reps):
        spoken = generate_turn(
            model,
            processor,
            build_conversation(
                audio_path=timed_source["path"],
                user_text=S2S_USER_TEXT,
            ),
            speaker=args.speaker,
            return_audio=True,
            wav_path=out_dir / f"timed_{index + 1}.wav",
        )
        thinker = generate_turn(
            model,
            processor,
            build_conversation(
                audio_path=timed_source["path"],
                user_text=S2S_USER_TEXT,
            ),
            speaker=args.speaker,
            return_audio=False,
            wav_path=None,
        )
        timed_rows.append(
            {
                "rep": index + 1,
                "thinker_latency_ms": thinker["latency_ms"],
                "talker_latency_ms": spoken["latency_ms"],
                "total_latency_ms": spoken["latency_ms"],
                "first_usable_text_ms": thinker["first_usable_text_ms"],
                "output_audio_duration_ms": spoken["audio_ms"],
                "rtf": spoken["rtf"],
                "resources": spoken["resources"],
                "text": spoken["text"],
            }
        )

    ethan = generate_turn(
        model,
        processor,
        build_conversation(
            audio_path=timed_source["path"],
            user_text=S2S_USER_TEXT,
        ),
        speaker="Ethan",
        return_audio=True,
        wav_path=out_dir / "out_reserve_ethan.wav",
    )

    tool_source = synthesize_input(
        timed_case["text"],
        out_dir / "in_tool.wav",
        model,
        processor,
        args.speaker,
    )
    tool_thinker = generate_turn(
        model,
        processor,
        build_conversation(
            audio_path=tool_source["path"],
            user_text=TOOL_AUDIO_HINT,
            include_tools=True,
        ),
        speaker=args.speaker,
        return_audio=False,
        wav_path=None,
    )
    tool_calls = tool_thinker["tool_calls"]
    if not tool_calls:
        tool_calls = [
            {
                "name": "check_availability",
                "arguments": {"date": "2026-09-19"},
            }
        ]
        tool_forced = True
    else:
        tool_forced = False
    tool_result = fake_fastify(tool_calls[0])
    tool_talker = generate_turn(
        model,
        processor,
        build_conversation(user_text=talker_user_text(tool_result["spoken"])),
        speaker=args.speaker,
        return_audio=True,
        wav_path=out_dir / "out_tool_talker.wav",
    )
    tool_success = (
        bool(tool_thinker["tool_calls"])
        and tool_result["ok"]
        and not tool_talker["quality"]["leaks_tool_json"]
        and (tool_talker["audio_ms"] or 0) > 300
    )

    thinker_latencies = [row["thinker_latency_ms"] for row in timed_rows]
    total_latencies = [row["total_latency_ms"] for row in timed_rows]
    rtf_values = [row["rtf"] for row in timed_rows if row["rtf"] is not None]
    french_acceptable = all(row["score"]["acceptable"] for row in french_rows)
    output_audio_ok = all((row["spoken"]["audio_ms"] or 0) > 300 for row in french_rows)
    peak_candidates = [
        after_load.get("peak_allocated_gb"),
        warmup.get("peak_allocated_gb"),
        *(row["spoken"].get("peak_allocated_gb") for row in french_rows),
        *(row["thinker"].get("peak_allocated_gb") for row in french_rows),
        ethan.get("peak_allocated_gb"),
        tool_talker.get("peak_allocated_gb"),
    ]
    numeric_peaks = [value for value in peak_candidates if isinstance(value, (int, float))]
    peak_vram = max(numeric_peaks) if numeric_peaks else None
    french_quality = "acceptable" if french_acceptable and output_audio_ok else "insufficient"
    go = decide_integration(
        oom=False,
        french_quality=french_quality,
        tool_call_success=tool_success,
        output_audio_ok=output_audio_ok,
        thinker_p50_ms=percentile(thinker_latencies, 0.5),
        total_p50_ms=percentile(total_latencies, 0.5),
    )
    return {
        "status": "measured",
        "model": args.model,
        "mode": args.mode,
        "speaker": args.speaker,
        "attn": args.attn,
        "hardware": after_load,
        "gpu_before_load": before,
        "model_load_time_ms": load_ms,
        "model_load_vram_gb": after_load.get("allocated_gb"),
        "idle_vram_gb": after_load.get("allocated_gb"),
        "peak_vram_gb": peak_vram,
        "warmup": {"latency_ms": warmup["latency_ms"], "audio_ms": warmup["audio_ms"]},
        "french_cases": french_rows,
        "timed_reps": timed_rows,
        "ethan_sample": {
            "text": ethan["text"],
            "audio_ms": ethan["audio_ms"],
            "quality": ethan["quality"],
            "wav": ethan["wav"],
        },
        "tool_path": {
            "input": tool_source,
            "thinker_text": tool_thinker["text"],
            "thinker_tool_calls": tool_thinker["tool_calls"],
            "forced_tool": tool_forced,
            "fake_fastify": tool_result,
            "talker_text": tool_talker["text"],
            "talker_leaks_json": tool_talker["quality"]["leaks_tool_json"],
            "talker_audio_ms": tool_talker["audio_ms"],
            "wav": tool_talker["wav"],
            "success": tool_success,
        },
        "summary": {
            "integration_allowed": go["integration_allowed"],
            "french_quality": french_quality,
            "tool_call_success": tool_success,
            "model_load_vram": after_load.get("allocated_gb"),
            "peak_vram": peak_vram,
            "total_latency_ms": percentile(total_latencies, 0.5),
            "total_latency_p50_ms": percentile(total_latencies, 0.5),
            "total_latency_p95_ms": percentile(total_latencies, 0.95),
            "thinker_latency_ms": percentile(thinker_latencies, 0.5),
            "thinker_latency_p50_ms": percentile(thinker_latencies, 0.5),
            "thinker_latency_p95_ms": percentile(thinker_latencies, 0.95),
            "talker_latency_ms": percentile(total_latencies, 0.5),
            "output_audio_ok": output_audio_ok,
            "rtf_p50": percentile(rtf_values, 0.5),
            "gpu_util_avg": _mean(
                [
                    value
                    for row in timed_rows
                    if row["resources"].get("gpu_util_avg") is not None
                    for value in [row["resources"]["gpu_util_avg"]]
                ]
            ),
            "cpu_util_avg": _mean(
                [
                    value
                    for row in timed_rows
                    if row["resources"].get("cpu_util_avg") is not None
                    for value in [row["resources"]["cpu_util_avg"]]
                ]
            ),
            "notes": go["notes"],
            "quality_go": go["quality_go"],
            "latency_justifies_streaming": go["latency_justifies_streaming"],
        },
    }


def build_report(measured: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    audit = compatibility_audit()
    summary = (measured or {}).get("summary") or {}
    oom = bool(error and "out of memory" in error.lower())
    if measured and summary:
        audit["vram"] = {**audit["vram"], "measured_peak_gb": summary.get("peak_vram")}
        audit["streaming"] = {
            **audit["streaming"],
            "measured_s2s_total_ms": summary.get("total_latency_ms"),
        }
        audit["function_calling"] = {
            **audit["function_calling"],
            "measured": {"tool_call_success": summary.get("tool_call_success")},
        }
        audit["french"] = {
            **audit["french"],
            "measured": {"french_quality": summary.get("french_quality")},
        }
    status = measured["status"] if measured else "not_measured"
    if error and status != "measured":
        go = decide_integration(
            oom=oom or True,
            french_quality="insufficient",
            tool_call_success=False,
            output_audio_ok=False,
            thinker_p50_ms=None,
            total_p50_ms=None,
        )
        summary = {
            "integration_allowed": False,
            "french_quality": "insufficient",
            "tool_call_success": False,
            "model_load_vram": None,
            "peak_vram": None,
            "total_latency_ms": None,
            "thinker_latency_ms": None,
            "talker_latency_ms": None,
            "output_audio_ok": False,
            "notes": go["notes"] + [error.splitlines()[0]],
        }
    return {
        "date": "2026-09-19",
        "goal": "Measure Qwen2.5-Omni-7B GPTQ-Int4 audio->audio on L4 before any Twilio refactor",
        "status": status,
        "error": error,
        "audit": audit,
        "run": measured,
        "integration_allowed": summary.get("integration_allowed", False),
        "french_quality": summary.get("french_quality"),
        "tool_call_success": summary.get("tool_call_success"),
        "model_load_vram": summary.get("model_load_vram"),
        "peak_vram": summary.get("peak_vram"),
        "total_latency_ms": summary.get("total_latency_ms"),
        "thinker_latency_ms": summary.get("thinker_latency_ms"),
        "talker_latency_ms": summary.get("talker_latency_ms"),
        "output_audio_ok": summary.get("output_audio_ok"),
        "notes": summary.get("notes") or [
            "Twilio stays disconnected until a measured L4 run exists."
        ],
        "go_nogo": {
            "integration_allowed": summary.get("integration_allowed", False),
            "reason": "; ".join(summary.get("notes") or ["not measured"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni INT4 S2S lab")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--mode",
        choices=("gpu_resident", "official_low_vram"),
        default="gpu_resident",
    )
    parser.add_argument("--speaker", default="Chelsie")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", default="benchmarks/results/omni_s2s_probe.json")
    parser.add_argument("--wav-dir", default="benchmarks/results/omni_s2s_wavs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", ""))
    measured: dict[str, Any] | None = None
    error: str | None = None
    if args.dry_run:
        measured = {
            "status": "not_measured",
            "reason": "dry-run on a host without loading weights",
            "hardware": _cuda_snapshot(),
        }
    else:
        try:
            measured = run_measured(args)
        except Exception as exc:
            error = f"{exc}\n{traceback.format_exc()}"
            measured = {"status": "error", "hardware": _cuda_snapshot()}
    report = build_report(measured, error)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "status": report["status"]}, ensure_ascii=True))
    return 0 if report["status"] in {"measured", "not_measured"} else 1


if __name__ == "__main__":
    sys.exit(main())
