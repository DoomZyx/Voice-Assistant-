"""TDD: barge-in confirme et epoch de generation. Pas de 150 ms magique."""

import pytest

from app.session.interrupt import (
    BargeInConfirmer,
    PlaybackEpoch,
    samples_to_ms,
    sweep_confirm_thresholds,
)


def test_samples_to_ms_at_8k():
    assert samples_to_ms(400, 8000) == 50.0
    assert samples_to_ms(1600, 8000) == 200.0


@pytest.mark.parametrize(
    ("confirm_ms", "burst_samples", "expect"),
    [
        (50, 400, True),
        (80, 400, False),
        (120, 400, False),
        (160, 400, False),
        (200, 400, False),
    ],
)
def test_50ms_burst_vs_confirm_thresholds(confirm_ms, burst_samples, expect):
    gate = BargeInConfirmer()
    assert gate.observe(burst_samples, True, confirm_ms=confirm_ms, sample_rate=8000) is expect


def test_silence_resets_confirmation():
    gate = BargeInConfirmer()
    gate.observe(800, True, confirm_ms=200, sample_rate=8000)
    assert gate.observe(160, False, confirm_ms=200, sample_rate=8000) is False
    assert gate.observe(800, True, confirm_ms=200, sample_rate=8000) is False


def test_continuous_speech_confirms_after_threshold():
    gate = BargeInConfirmer()
    confirmed = False
    for _ in range(10):
        confirmed = gate.observe(160, True, confirm_ms=160, sample_rate=8000)
    assert confirmed is True
    assert gate.speech_ms(8000) >= 160


def test_epoch_drops_stale_generation():
    epoch = PlaybackEpoch()
    first = epoch.current
    assert epoch.is_current(first) is True
    epoch.bump()
    assert epoch.is_current(first) is False
    assert epoch.is_current(epoch.current) is True


def test_confirm_sweep_does_not_treat_150_as_magic():
    rows = {row["confirm_ms"]: row for row in sweep_confirm_thresholds()}
    assert rows[50]["burst_50ms_interrupts"] is True
    assert rows[160]["burst_50ms_interrupts"] is False
    assert rows[160]["speech_200ms_interrupts"] is True
    assert 150 not in rows
