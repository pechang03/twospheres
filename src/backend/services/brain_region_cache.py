"""Brain Region Cache with LRU eviction and smart prefetching.

Implements advanced caching for brain region tensors with:
- LRU cache with k=20 capacity (20 out of 380 regions)
- Smart prefetching of r-IDS neighbors (k'=2 levels)
- Background prefetch worker
- Target hit rate: 80-90%
- Hit latency: <1ms
- Miss latency: <100ms (with prefetch)

Reference: docs/designs/yada-hierarchical-brain-model/BRAIN_QEC_CACHE_CROSSTRAINING.md
"""

import asyncio
import time
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegionTensor:
    """Placeholder for brain region tensor data."""

    region_name: str
    region_id: Optional[int] = None
    data: Any = None
    rids_connections: List[str] = field(default_factory=list)
    features: Optional[Any] = None
    syndrome_history: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def compute_syndrome(self, predicted: Any, actual: Any) -> float:
        """Compute syndrome (prediction error) between predicted and actual.

        Args:
            predicted: Predicted tensor values
            actual: Actual tensor values

        Returns:
            syndrome: Prediction error magnitude (0-1 scale)
        """
        if predicted is None or actual is None:
            return 1.0

        # Simple L2 norm-based syndrome
        import numpy as np
        if isinstance(predicted, (list, tuple)):
            predicted = np.array(predicted)
        if isinstance(actual, (list, tuple)):
            actual = np.array(actual)

        try:
            diff = np.linalg.norm(predicted - actual)
            # Normalize to 0-1 range
            syndrome = min(diff / 100.0, 1.0)  # 100 is max expected error
            return syndrome
        except (TypeError, ValueError):
            return 0.5  # Default middle value


class BrainRegionCache:
    """LRU cache for brain region tensors with smart prefetching.

    Features:
    - LRU eviction (least recently used regions evicted first)
    - Smart prefetching of r-IDS neighbors
    - Background prefetch worker for non-blocking loads
    - Hit/miss tracking and statistics
    - O(1) cache lookups, O(k) prefetch scheduling

    Performance targets:
    - Hit latency: <1ms (dict lookup in memory)
    - Miss latency: <100ms (with prefetch reducing future misses)
    - Target hit rate: 80-90%
    """

    def __init__(
        self,
        capacity: int = 20,
        prefetch_depth: int = 2,
        enable_prefetch: bool = True,
    ):
        """Initialize cache with LRU eviction.

        Args:
            capacity: Max regions in cache (k=20 default, 20/380 = 5.3%)
            prefetch_depth: r-IDS neighborhood depth for prefetch (k'=2)
            enable_prefetch: If True, enable background prefetching
        """
        self.capacity = capacity
        self.prefetch_depth = prefetch_depth
        self.enable_prefetch = enable_prefetch

        # LRU cache: OrderedDict maintains insertion order
        # Most recently used items are at the end
        self.cache: OrderedDict[str, RegionTensor] = OrderedDict()

        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.access_times: Dict[str, float] = {}  # Track access latencies

        # Prefetch state
        self.prefetch_queue: Optional[asyncio.Queue] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        self._prefetch_running = False
        self._lock: Optional[asyncio.Lock] = None
        self._initialized = False

    def _ensure_async_resources(self):
        """Ensure async resources are initialized."""
        if not self._initialized:
            try:
                loop = asyncio.get_running_loop()
                if self.prefetch_queue is None:
                    self.prefetch_queue = asyncio.Queue()
                if self._lock is None:
                    self._lock = asyncio.Lock()
                self._initialized = True
            except RuntimeError:
                # No running loop yet, will be initialized on first async call
                pass

    async def get_or_load(self, region_name: str) -> RegionTensor:
        """Get region from cache or load from database.

        Hit path: O(1) dictionary lookup + move to end
        Miss path: Load from DB + schedule prefetch of neighbors

        Args:
            region_name: Name of brain region (e.g., "V1", "V2")

        Returns:
            tensor: RegionTensor containing region data

        Performance:
            - Cache hit: <1ms (dict lookup + OrderedDict.move_to_end)
            - Cache miss: <100ms (simulated DB load, prefetch scheduled async)
        """
        self._ensure_async_resources()
        start_time = time.time()

        async with self._lock:
            # Cache hit
            if region_name in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(region_name)  # Mark as recently used
                hit_latency = (time.time() - start_time) * 1000  # ms
                self.access_times[region_name] = hit_latency
                logger.debug(
                    f"Cache HIT: {region_name} (latency: {hit_latency:.2f}ms)"
                )
                return self.cache[region_name]

            # Cache miss
            self.cache_misses += 1
            logger.debug(f"Cache MISS: {region_name}")

            # Load from database (simulated)
            tensor = await self._load_from_database(region_name)

            # Add to cache (evict LRU if needed)
            if len(self.cache) >= self.capacity:
                evicted_name, _ = self.cache.popitem(last=False)  # Remove LRU
                logger.debug(f"LRU Eviction: {evicted_name}")

            self.cache[region_name] = tensor

            # Schedule prefetch of r-IDS neighbors if enabled
            if self.enable_prefetch and tensor.rids_connections:
                await self._schedule_prefetch(tensor.rids_connections)

        miss_latency = (time.time() - start_time) * 1000  # ms
        self.access_times[region_name] = miss_latency
        logger.debug(f"Cache MISS latency: {region_name} ({miss_latency:.2f}ms)")

        return tensor

    async def _load_from_database(self, region_name: str) -> RegionTensor:
        """Load region tensor from database (placeholder).

        In production, this would query PostgreSQL/MongoDB.
        Currently simulates 50ms database latency.

        Args:
            region_name: Region to load

        Returns:
            tensor: Loaded RegionTensor
        """
        # Simulate database load latency
        await asyncio.sleep(0.05)  # 50ms simulated DB latency

        # Create mock tensor with simulated r-IDS connections
        tensor = RegionTensor(
            region_name=region_name,
            region_id=hash(region_name) % 380,
            data={"placeholder": True},
        )

        # Simulate r-IDS neighborhood connections (k'=2 levels)
        # In production, would come from r-IDS algorithm service
        tensor.rids_connections = self._get_mock_rids_neighbors(region_name)

        logger.debug(
            f"Loaded {region_name} from DB with {len(tensor.rids_connections)} r-IDS neighbors"
        )
        return tensor

    def _get_mock_rids_neighbors(self, region_name: str) -> List[str]:
        """Get mock r-IDS neighbors for region.

        In production, would call merge2docs r-IDS algorithm service.
        Currently returns deterministic neighbors for testing.

        Args:
            region_name: Region name

        Returns:
            neighbors: List of neighboring region names
        """
        # Simple mock: visual regions connected to each other
        mock_neighbors = {
            "V1": ["V2", "V3", "V4"],
            "V2": ["V1", "V3", "V4", "MT"],
            "V3": ["V1", "V2", "V4"],
            "V4": ["V1", "V2", "V3", "MT", "MST"],
            "MT": ["V2", "V4", "MST", "MSTd"],
            "MST": ["V4", "MT", "MSTd"],
            "MSTd": ["MT", "MST"],
            "M1": ["PM", "S1"],
            "PM": ["M1", "S1", "PFC"],
            "S1": ["M1", "PM", "S2"],
            "S2": ["S1", "PM"],
            "PFC": ["PM", "V4", "MT"],
        }
        return mock_neighbors.get(region_name, [])

    async def _schedule_prefetch(self, neighbors: List[str]):
        """Schedule background prefetching of r-IDS neighbors.

        Adds neighbors to prefetch queue and starts background worker
        if not already running.

        Args:
            neighbors: List of neighboring region names to prefetch
        """
        if not self.enable_prefetch:
            return

        self._ensure_async_resources()

        # Add neighbors to prefetch queue (if not already in cache)
        for neighbor in neighbors:
            if neighbor not in self.cache:
                try:
                    self.prefetch_queue.put_nowait(neighbor)
                except asyncio.QueueFull:
                    logger.warning(f"Prefetch queue full, skipping {neighbor}")

        # Start prefetch worker if not running
        if not self._prefetch_running:
            self._prefetch_task = asyncio.create_task(self._prefetch_worker())

    async def _prefetch_worker(self):
        """Background worker for prefetching regions.

        Continuously processes prefetch queue, loading regions into cache
        before they're requested. Stops when queue is empty.

        This is a fire-and-forget worker that runs asynchronously while
        main cache operations continue.
        """
        self._prefetch_running = True
        logger.debug("Prefetch worker started")

        try:
            while True:
                try:
                    # Get next region to prefetch with timeout
                    region_name = await asyncio.wait_for(
                        self.prefetch_queue.get(), timeout=0.5
                    )

                    # Skip if already in cache
                    if region_name not in self.cache:
                        logger.debug(f"Prefetch: {region_name}")
                        # Load without updating statistics (it's a prefetch)
                        tensor = await self._load_from_database(region_name)

                        # Add to cache if space available
                        async with self._lock:
                            if len(self.cache) < self.capacity:
                                self.cache[region_name] = tensor
                                logger.debug(
                                    f"Prefetched {region_name} into cache"
                                )

                    self.prefetch_queue.task_done()

                except asyncio.TimeoutError:
                    # Queue empty or timed out
                    break
                except Exception as e:
                    logger.error(f"Prefetch worker error: {e}")
                    break

        finally:
            self._prefetch_running = False
            logger.debug("Prefetch worker stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            stats: Dictionary containing:
                - capacity: Max cache size
                - size: Current cache size
                - hits: Total cache hits
                - misses: Total cache misses
                - hit_rate: Hit rate percentage (0-1)
                - avg_hit_latency_ms: Average latency for hits
                - avg_miss_latency_ms: Average latency for misses
                - cached_regions: List of currently cached regions
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        # Calculate average latencies
        hit_latencies = [
            self.access_times.get(region, 0.5)
            for region in self.cache.keys()
            if region in self.access_times
        ]
        avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0

        all_latencies = list(self.access_times.values())
        avg_miss_latency = (
            sum(all_latencies) / len(all_latencies) if all_latencies else 50
        )

        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.capacity,
            "avg_hit_latency_ms": avg_hit_latency,
            "avg_miss_latency_ms": avg_miss_latency,
            "cached_regions": list(self.cache.keys()),
        }

    async def clear(self):
        """Clear cache and stop prefetch worker."""
        self._ensure_async_resources()

        if self._lock:
            async with self._lock:
                self.cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0
                self.access_times.clear()
        else:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.access_times.clear()

        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"BrainRegionCache(capacity={stats['capacity']}, "
            f"size={stats['size']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
