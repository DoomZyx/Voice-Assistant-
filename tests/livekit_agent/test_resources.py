from livekit_agent.observability.resources import SessionRegistry, sample_resources


def test_session_registry_tracks_concurrency():
    registry = SessionRegistry()
    assert registry.count() == 0
    assert registry.register("a") == 1
    assert registry.register("b") == 2
    assert registry.register("a") == 2
    assert registry.release("a") == 1
    assert registry.release("missing") == 1
    assert registry.release("b") == 0


def test_sample_resources_has_required_keys():
    sample = sample_resources(concurrent_sessions=3)
    assert sample["concurrent_sessions"] == 3
    assert "cpu_percent" in sample
    assert "ram_rss_mb" in sample
    assert "gpu_vram_mb" in sample
    assert "gpu_utilization" in sample
