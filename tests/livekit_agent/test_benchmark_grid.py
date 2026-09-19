from benchmarks.livekit.concurrent_calls import CONCURRENCY_LEVELS


def test_concurrency_grid_matches_plan():
    assert CONCURRENCY_LEVELS == (1, 5, 10, 15, 20, 30, 40)
