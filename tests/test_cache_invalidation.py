from src.serving.cache import RecommendationCache


def test_cache_invalidate_one_limit():
    cache = RecommendationCache(url="redis://nonexistent-host:6379/0")
    cache.set(7, 5, [{"item_id": 1}])
    cache.set(7, 10, [{"item_id": 2}])
    cache.invalidate(7, limit=5)
    assert cache.get(7, 5) is None
    assert cache.get(7, 10) == [{"item_id": 2}]


def test_cache_invalidate_all_limits_for_user_only():
    cache = RecommendationCache(url="redis://nonexistent-host:6379/0")
    cache.set(7, 5, [{"item_id": 1}])
    cache.set(7, 10, [{"item_id": 2}])
    cache.set(8, 5, [{"item_id": 3}])
    cache.invalidate(7)
    assert cache.get(7, 5) is None
    assert cache.get(7, 10) is None
    assert cache.get(8, 5) == [{"item_id": 3}]


def test_cache_rejects_nonpositive_ttl():
    import pytest
    with pytest.raises(ValueError):
        RecommendationCache(ttl=0)
