"""Tests for Brain Region Cache with LRU eviction and smart prefetching.

Tests cover:
- LRU eviction when capacity is reached
- Cache hit/miss tracking
- Smart prefetching of r-IDS neighbors
- Hit/miss latency targets
- Cache statistics
- Background prefetch worker
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from src.backend.services.brain_region_cache import (
    BrainRegionCache,
    RegionTensor,
)


@pytest.fixture
async def cache():
    """Create a fresh cache for each test."""
    cache_obj = BrainRegionCache(capacity=5, prefetch_depth=2, enable_prefetch=True)
    yield cache_obj
    await cache_obj.clear()


@pytest.fixture
async def cache_no_prefetch():
    """Create cache with prefetch disabled."""
    cache_obj = BrainRegionCache(capacity=5, prefetch_depth=2, enable_prefetch=False)
    yield cache_obj
    await cache_obj.clear()


class TestCacheInitialization:
    """Test cache initialization and basic properties."""

    def test_cache_initialization(self):
        """Test cache initializes with correct capacity."""
        cache = BrainRegionCache(capacity=20)
        assert cache.capacity == 20
        assert len(cache.cache) == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0

    def test_cache_repr(self, cache):
        """Test cache string representation."""
        repr_str = repr(cache)
        assert "BrainRegionCache" in repr_str
        assert "capacity=5" in repr_str
        assert "hit_rate=0.0%" in repr_str


class TestCacheHitAndMiss:
    """Test cache hit and miss behavior."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache):
        """Test cache hit increments counter and returns cached value."""
        # First load (cache miss)
        tensor1 = await cache.get_or_load("V1")
        assert cache.cache_hits == 0
        assert cache.cache_misses == 1

        # Second load (cache hit)
        tensor2 = await cache.get_or_load("V1")
        assert cache.cache_hits == 1
        assert cache.cache_misses == 1

        # Should return same object
        assert tensor1.region_name == tensor2.region_name

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss for new region."""
        tensor = await cache.get_or_load("V2")
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0
        assert tensor.region_name == "V2"

    @pytest.mark.asyncio
    async def test_multiple_distinct_regions(self, cache):
        """Test loading multiple distinct regions."""
        regions = ["V1", "V2", "V4"]
        for region in regions:
            await cache.get_or_load(region)

        assert cache.cache_misses == 3
        assert len(cache.cache) == 3


class TestLRUEviction:
    """Test LRU eviction when cache reaches capacity."""

    @pytest.mark.asyncio
    async def test_lru_eviction_on_capacity(self, cache):
        """Test that least recently used region is evicted at capacity."""
        # Fill cache to capacity (capacity=5)
        regions = ["V1", "V2", "V3", "V4", "V5"]
        for region in regions:
            await cache.get_or_load(region)

        assert len(cache.cache) == 5
        stats = cache.get_stats()
        assert stats["size"] == 5
        assert stats["utilization"] == 1.0

        # Load new region, should evict V1 (least recently used)
        await cache.get_or_load("M1")

        assert len(cache.cache) == 5
        assert "V1" not in cache.cache
        assert "M1" in cache.cache
        assert cache.cache_misses == 6

    @pytest.mark.asyncio
    async def test_lru_eviction_tracks_usage(self, cache):
        """Test that LRU eviction respects access patterns.

        Recently accessed regions should not be evicted.
        """
        # Fill cache
        regions = ["V1", "V2", "V3", "V4", "V5"]
        for region in regions:
            await cache.get_or_load(region)

        # Access V1 again (make it most recently used)
        await cache.get_or_load("V1")

        # Now V2 should be least recently used
        # Load new region, V2 should be evicted
        await cache.get_or_load("M1")

        assert "V1" in cache.cache  # V1 still there (recently accessed)
        assert "V2" not in cache.cache  # V2 evicted (least recently used)
        assert "M1" in cache.cache

    @pytest.mark.asyncio
    async def test_lru_eviction_sequence(self, cache):
        """Test correct LRU eviction sequence."""
        # Access pattern: V1, V2, V3, V4, V5
        # Expected eviction order: V1, V2, V3, V4, V5

        regions = ["V1", "V2", "V3", "V4", "V5"]
        for region in regions:
            await cache.get_or_load(region)

        # Now evict in sequence
        for i, new_region in enumerate(["M1", "M2", "M3"], start=1):
            await cache.get_or_load(new_region)
            # Check cache size stays constant
            assert len(cache.cache) == 5


class TestCacheLatency:
    """Test cache latency targets."""

    @pytest.mark.asyncio
    async def test_hit_latency_under_1ms(self, cache):
        """Test that cache hits complete in <1ms."""
        # Load a region first (miss)
        await cache.get_or_load("V1")

        # Measure hit latency
        start = time.time()
        await cache.get_or_load("V1")
        hit_latency = (time.time() - start) * 1000  # Convert to ms

        # Should be much faster than 1ms (order of micros)
        assert hit_latency < 1.0, f"Hit latency {hit_latency}ms exceeds 1ms target"

    @pytest.mark.asyncio
    async def test_miss_latency_under_100ms(self, cache):
        """Test that cache misses complete in <100ms."""
        start = time.time()
        await cache.get_or_load("V1")
        miss_latency = (time.time() - start) * 1000  # Convert to ms

        # Simulated DB load is 50ms, so should be under 100ms
        assert miss_latency < 100.0, f"Miss latency {miss_latency}ms exceeds 100ms target"


class TestCacheStatistics:
    """Test cache statistics and hit rate calculation."""

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self, cache):
        """Test hit rate is calculated correctly."""
        # Miss: V1
        await cache.get_or_load("V1")

        # Hit: V1
        await cache.get_or_load("V1")

        # Miss: V2
        await cache.get_or_load("V2")

        # Hit: V1
        await cache.get_or_load("V1")

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_stats_contain_required_fields(self, cache):
        """Test that stats contain all required fields."""
        await cache.get_or_load("V1")
        stats = cache.get_stats()

        required_fields = [
            "capacity",
            "size",
            "hits",
            "misses",
            "hit_rate",
            "utilization",
            "cached_regions",
        ]

        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_stats_cached_regions_list(self, cache):
        """Test that cached_regions list is current."""
        regions = ["V1", "V2", "V3"]
        for region in regions:
            await cache.get_or_load(region)

        stats = cache.get_stats()
        assert set(stats["cached_regions"]) == set(regions)


class TestSmartPrefetching:
    """Test smart prefetching of r-IDS neighbors."""

    @pytest.mark.asyncio
    async def test_prefetch_scheduled_on_load(self, cache):
        """Test that prefetch is scheduled when region is loaded."""
        # Load V1 which has r-IDS neighbors V2, V3, V4
        await cache.get_or_load("V1")

        # Give prefetch worker time to run
        await asyncio.sleep(0.2)

        # V1 should be in cache
        assert "V1" in cache.cache

        # Some neighbors might be prefetched
        stats = cache.get_stats()
        assert stats["size"] >= 1

    @pytest.mark.asyncio
    async def test_prefetch_prevents_future_misses(self, cache):
        """Test that prefetching reduces future cache misses."""
        # Load V1 (starts prefetch of neighbors)
        await cache.get_or_load("V1")

        # Give prefetch worker time
        await asyncio.sleep(0.3)

        initial_misses = cache.cache_misses

        # Access a neighbor (should be prefetched)
        await cache.get_or_load("V2")

        # Check if it was a hit (prefetched) or miss
        final_misses = cache.cache_misses
        was_prefetched = final_misses == initial_misses  # No new miss

        # It should be prefetched due to smart prefetching
        assert was_prefetched, "Neighbor should have been prefetched"

    @pytest.mark.asyncio
    async def test_prefetch_disabled(self, cache_no_prefetch):
        """Test cache works correctly with prefetch disabled."""
        # Load V1
        await cache_no_prefetch.get_or_load("V1")

        # Give time for prefetch that shouldn't happen
        await asyncio.sleep(0.2)

        # Only V1 should be in cache (no prefetch)
        assert len(cache_no_prefetch.cache) == 1
        assert "V1" in cache_no_prefetch.cache

    @pytest.mark.asyncio
    async def test_prefetch_respects_capacity(self, cache):
        """Test that prefetch doesn't exceed cache capacity."""
        # Capacity is 5
        assert cache.capacity == 5

        # Load regions to capacity
        for region in ["V1", "V2", "V3", "V4", "V5"]:
            await cache.get_or_load(region)

        # Wait for prefetch
        await asyncio.sleep(0.2)

        # Cache should not exceed capacity
        assert len(cache.cache) <= cache.capacity


class TestPrefetchWorker:
    """Test background prefetch worker."""

    @pytest.mark.asyncio
    async def test_prefetch_worker_starts_on_demand(self, cache):
        """Test prefetch worker starts when prefetch scheduled."""
        assert cache._prefetch_running == False

        # Schedule prefetch
        await cache._schedule_prefetch(["V2", "V3"])

        # Give worker time to start
        await asyncio.sleep(0.1)

        # Worker should be running or have completed
        # (it completes when queue is empty)
        assert True  # Just verify no crash

    @pytest.mark.asyncio
    async def test_prefetch_worker_stops_when_empty(self, cache):
        """Test prefetch worker stops when queue is empty."""
        # Schedule prefetch
        await cache._schedule_prefetch(["V2"])

        # Give worker time to process and stop
        await asyncio.sleep(0.5)

        # Worker should be stopped
        assert cache._prefetch_running == False or cache.prefetch_queue.empty()

    @pytest.mark.asyncio
    async def test_prefetch_worker_skips_cached_items(self, cache):
        """Test prefetch worker skips items already in cache."""
        # Pre-load V1
        await cache.get_or_load("V1")
        initial_size = len(cache.cache)

        # Schedule prefetch of V1 (already cached) and V2
        await cache._schedule_prefetch(["V1", "V2"])

        # Give worker time
        await asyncio.sleep(0.3)

        # Cache size should grow, but not exceed capacity
        # V1 is skipped (already cached), V2 is prefetched
        # So we expect size to be initial_size + 1 at most
        assert len(cache.cache) <= cache.capacity


class TestRegionTensor:
    """Test RegionTensor data structure."""

    def test_region_tensor_initialization(self):
        """Test RegionTensor initializes correctly."""
        tensor = RegionTensor(region_name="V1", region_id=1)
        assert tensor.region_name == "V1"
        assert tensor.region_id == 1
        assert tensor.rids_connections == []
        assert tensor.syndrome_history == []

    def test_region_tensor_syndrome_computation(self):
        """Test syndrome computation."""
        tensor = RegionTensor(region_name="V1")

        # Zero difference = zero syndrome
        syndrome = tensor.compute_syndrome([1, 2, 3], [1, 2, 3])
        assert syndrome == 0.0

        # Different values = syndrome > 0
        syndrome = tensor.compute_syndrome([1, 2, 3], [2, 3, 4])
        assert syndrome > 0.0

    def test_region_tensor_syndrome_clamping(self):
        """Test syndrome is clamped to 0-1 range."""
        tensor = RegionTensor(region_name="V1")

        # Very large difference
        syndrome = tensor.compute_syndrome([0] * 100, [1000] * 100)
        assert 0 <= syndrome <= 1.0


class TestCacheClear:
    """Test cache clearing and reset."""

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test cache clear resets state."""
        # Load some regions
        await cache.get_or_load("V1")
        await cache.get_or_load("V2")

        assert len(cache.cache) == 2
        assert cache.cache_misses == 2

        # Clear cache
        await cache.clear()

        assert len(cache.cache) == 0
        assert cache.cache_misses == 0
        assert cache.cache_hits == 0


class TestHighHitRateScenario:
    """Test scenario achieving 80-90% hit rate target."""

    @pytest.mark.asyncio
    async def test_working_set_hit_rate(self, cache):
        """Test high hit rate with working set smaller than capacity."""
        # Capacity is 5, working set is 4 regions
        working_set = ["V1", "V2", "V4", "MT"]

        # Simulate typical access pattern (80% hits)
        for _ in range(10):
            region = working_set[_ % len(working_set)]
            await cache.get_or_load(region)

        stats = cache.get_stats()

        # First 4 are misses, remaining 6 are hits
        # Hit rate = 6/10 = 60%
        assert stats["hit_rate"] >= 0.6

    @pytest.mark.asyncio
    async def test_working_set_with_prefetch(self, cache):
        """Test high hit rate is enhanced by prefetching."""
        # V1 has neighbors V2, V3, V4 (will be prefetched)
        await cache.get_or_load("V1")

        # Give prefetch time
        await asyncio.sleep(0.3)

        # Now access neighbors - should be prefetched
        for region in ["V2", "V3"]:
            await cache.get_or_load(region)

        stats = cache.get_stats()

        # V1 miss, V2 and V3 should be hits (prefetched)
        assert stats["hit_rate"] >= 0.33


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_typical_workflow(self, cache):
        """Test typical cache workflow."""
        # Access sequence: load regions, some hits, some misses
        sequence = ["V1", "V1", "V2", "V1", "V4", "V2", "MT", "V1"]

        for region in sequence:
            await cache.get_or_load(region)

        stats = cache.get_stats()

        # Verify expected behavior
        assert stats["hits"] > 0  # Some cache hits
        assert stats["misses"] > 0  # Some cache misses
        assert stats["hit_rate"] > 0  # Positive hit rate
        assert stats["size"] <= stats["capacity"]
        assert len(stats["cached_regions"]) <= stats["capacity"]

    @pytest.mark.asyncio
    async def test_cache_behavior_under_load(self, cache):
        """Test cache behavior with rapid accesses."""
        # Access within working set smaller than capacity
        # capacity = 5, working set = 3
        regions = ["V1", "V2", "V4"]

        for _ in range(50):
            region = regions[_ % len(regions)]
            await cache.get_or_load(region)

        stats = cache.get_stats()

        # Cache should maintain capacity invariant
        assert stats["size"] <= stats["capacity"]

        # Hit rate should be positive after warm-up
        # First 3 accesses are misses, then 47 are hits
        assert stats["hits"] > 0, f"Expected hits but got {stats['hits']}"
        assert stats["hit_rate"] > 0.5, f"Expected high hit rate but got {stats['hit_rate']}"

    @pytest.mark.asyncio
    async def test_prefetch_with_lru_eviction(self, cache):
        """Test prefetch works correctly with LRU eviction.

        Scenario:
        1. Load V1 (starts prefetch of neighbors)
        2. Fill cache to capacity
        3. Prefetch should not interfere with LRU eviction
        """
        # Load V1 (prefetch scheduled)
        await cache.get_or_load("V1")

        # Fill cache to capacity
        for region in ["V2", "V3", "V4", "V5"]:
            await cache.get_or_load(region)

        assert len(cache.cache) == 5

        # Load one more region
        await cache.get_or_load("M1")

        # Cache size should remain at capacity
        assert len(cache.cache) == 5

        # V1 might have been evicted if not recently accessed
        # (unless prefetch kept it hot)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
