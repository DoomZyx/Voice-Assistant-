import numpy as np

from livekit_agent.stt.pcm import frames_to_pcm16, resample_to_16k
from livekit_agent.tts.pcm import pcm16_chunks


def test_frames_to_pcm16_concatenates_mono():
    left = np.array([1, 2, 3], dtype=np.int16)
    right = np.array([4, 5], dtype=np.int16)
    pcm, rate = frames_to_pcm16(
        [
            (left.tobytes(), 16000, 1),
            (right.tobytes(), 16000, 1),
        ]
    )
    assert rate == 16000
    assert pcm.tolist() == [1, 2, 3, 4, 5]


def test_resample_to_16k_is_noop_when_already_16k():
    pcm = np.arange(16, dtype=np.int16)
    out = resample_to_16k(pcm, 16000)
    assert out.dtype == np.int16
    assert out.tolist() == pcm.tolist()


def test_frames_to_pcm16_downmixes_48k_stereo():
    left = np.full(48, 10, dtype=np.int16)
    right = np.full(48, 30, dtype=np.int16)
    stereo = np.empty(96, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right
    pcm, rate = frames_to_pcm16([(stereo.tobytes(), 48000, 2)])
    assert rate == 48000
    assert pcm.tolist() == [20] * 48
    out = resample_to_16k(pcm, 48000)
    assert out.dtype == np.int16
    assert out.size == 16


def test_pcm16_chunks_split_by_frame_ms():
    pcm = np.arange(48, dtype=np.int16)
    chunks = list(pcm16_chunks(pcm, sample_rate=24000, frame_ms=1))
    assert len(chunks) == 2
    assert np.frombuffer(chunks[0], dtype=np.int16).size == 24
    assert np.frombuffer(chunks[1], dtype=np.int16).size == 24
