import numpy as np

from app.audio.preprocess import apply_preprocess, peak_normalize_pcm16
from app.config import Settings
from app.stt.faster_whisper import whisper_transcribe_options


def test_peak_normalize_lowers_clipping_signal():
    pcm = np.array([32767, -32767, 30000, 0], dtype=np.int16)
    out = peak_normalize_pcm16(pcm, peak=0.72)
    assert int(np.max(np.abs(out))) <= int(32767 * 0.72) + 1
    assert out[3] == 0
    assert np.sign(out[0]) == 1
    assert np.sign(out[1]) == -1


def test_peak_normalize_leaves_quiet_audio():
    pcm = np.array([800, -800, 0], dtype=np.int16)
    out = peak_normalize_pcm16(pcm, peak=0.72)
    assert np.array_equal(out, pcm)


def test_agc_raises_quiet_telephone_speech():
    quiet = np.full(1600, 400, dtype=np.int16)
    boosted = apply_preprocess(quiet, "agc")
    assert int(np.max(np.abs(boosted))) > int(np.max(np.abs(quiet)))


def test_whisper_options_favor_capture_over_speed():
    options = whisper_transcribe_options(
        Settings(voice_eager_load=False, whisper_beam_size=5, whisper_language="fr")
    )
    assert options["beam_size"] == 5
    assert options["condition_on_previous_text"] is False
    assert options["vad_filter"] is False
    assert options["language"] == "fr"
