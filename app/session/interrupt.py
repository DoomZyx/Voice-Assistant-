"""Barge-in confirme + epoch de lecture. Le seuil n'est pas une constante optimale."""

from __future__ import annotations


def samples_to_ms(samples: int, sample_rate: int) -> float:
    rate = max(int(sample_rate or 0), 1)
    return 1000.0 * max(int(samples or 0), 0) / rate


class BargeInConfirmer:
    """Accumule la parole continue. Un claquement court ne doit pas clear."""

    def __init__(self) -> None:
        self.samples = 0

    def reset(self) -> None:
        self.samples = 0

    def speech_ms(self, sample_rate: int) -> float:
        return samples_to_ms(self.samples, sample_rate)

    def observe(
        self,
        sample_count: int,
        is_speech: bool,
        *,
        confirm_ms: int,
        sample_rate: int,
    ) -> bool:
        if not is_speech or sample_count <= 0:
            self.reset()
            return False
        self.samples += int(sample_count)
        return self.speech_ms(sample_rate) >= max(int(confirm_ms or 0), 0)


class PlaybackEpoch:
    """Invalide les frames TTS d'une generation precedente apres barge-in."""

    def __init__(self) -> None:
        self.current = 1

    def bump(self) -> int:
        self.current += 1
        return self.current

    def is_current(self, generation: int) -> bool:
        return generation == self.current


CONFIRM_CANDIDATES_MS = (50, 80, 120, 160, 200, 250)


def sweep_confirm_thresholds(
    *,
    sample_rate: int = 8000,
    burst_ms: int = 50,
    speech_ms: int = 200,
    noise_ms: int = 250,
    thresholds: tuple[int, ...] = CONFIRM_CANDIDATES_MS,
) -> list[dict[str, bool | int]]:
    """Compare les seuils. N'impose pas 150 ms comme optimum."""
    rows: list[dict[str, bool | int]] = []
    burst_samples = int(sample_rate * burst_ms / 1000)
    speech_samples = int(sample_rate * speech_ms / 1000)
    noise_samples = int(sample_rate * noise_ms / 1000)
    for confirm_ms in thresholds:
        burst = BargeInConfirmer().observe(
            burst_samples, True, confirm_ms=confirm_ms, sample_rate=sample_rate
        )
        speech_gate = BargeInConfirmer()
        speech_ok = speech_gate.observe(
            speech_samples, True, confirm_ms=confirm_ms, sample_rate=sample_rate
        )
        noise = BargeInConfirmer().observe(
            noise_samples, True, confirm_ms=confirm_ms, sample_rate=sample_rate
        )
        rows.append(
            {
                "confirm_ms": confirm_ms,
                "burst_50ms_interrupts": burst,
                "speech_200ms_interrupts": speech_ok,
                "sustained_noise_250ms_interrupts": noise,
            }
        )
    return rows
