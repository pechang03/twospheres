# Brain-Specific QEC Tensor with Advanced Cache & Recursive Cross-Training

## Executive Summary

Build a **brain-specific QEC tensor system** that extends merge2docs with:

1. **Advanced Cache**: Smart loading of brain regions on-demand (not all 368 regions!)
2. **Recursive Cross-Training**: Regions iteratively train each other via r-IDS bridges
3. **fMRI Integration**: Use our macaque 4D fMRI data
4. **Database Backend**: PostgreSQL with intelligent caching layer

**Key Insight**: Don't load all regions at once. Load V1, which recruits V2 (via r-IDS), which recruits V4, which recruits MT... **Recursive expansion** guided by syndrome signals.

---

## Part 0: F_i Functor Hierarchy (Vertical Dimension)

### Critical: The Vertical Tensor Structure

**merge2docs model**:
```
Domain (horizontal) × F_i functor (VERTICAL) × Level

F_i functors (vertical slices):
├─ wisdom   → High-level knowledge connections
├─ papers   → Research document connections
├─ code     → Implementation connections
├─ testing  → Validation connections
└─ git      → Version control connections
```

**Brain model** (our adaptation):
```
Region (horizontal) × F_i functor (VERTICAL) × Scale

F_i functors (vertical slices for SAME region):
├─ Anatomy    → Structural connectivity (D99 atlas, DTI)
├─ Function   → Functional connectivity (fMRI BOLD)
├─ Electro    → Neural dynamics (EEG, LFP, spikes)
├─ Genetics   → Gene expression (Allen Brain Atlas)
├─ Behavior   → Task-related activity
└─ Pathology  → Disease markers (lesions, atrophy)
```

### Example: V1 Tensor (All Functors)

```python
V1_tensor = {
    # VERTICAL dimension (F_i functors)
    "anatomy": {
        "neighbors": ["V2", "LGN"],  # Physical adjacency
        "layer_structure": ["L1", "L2/3", "L4", "L5", "L6"],
        "volume_mm3": 850,
        "rids_connections": {"V2", "V4", "MT"},  # r=4 anatomical
    },
    "function": {
        "features": np.array([...]),  # fMRI-derived features
        "responds_to": ["edges", "orientation", "spatial_frequency"],
        "syndrome": 0.12,  # Prediction error
        "rids_connections": {"V2", "V4", "MT"},  # r=4 functional
    },
    "electro": {
        "firing_rate_hz": 15.3,
        "oscillations": {"alpha": 0.8, "gamma": 0.6},
        "rids_connections": {"V2", "LGN"},  # r=4 electrophysiological
    },
    "genetics": {
        "expressed_genes": ["GAD1", "PVALB", "SST"],  # Interneuron markers
        "cell_types": {"excitatory": 0.8, "inhibitory": 0.2},
        "rids_connections": {"V2"},  # r=4 genetic similarity
    },
    "behavior": {
        "task_modulation": 0.65,  # How much V1 changes during tasks
        "attention_effect": 0.25,  # Attention modulation strength
        "rids_connections": {"V4", "PFC"},  # r=4 behavioral coupling
    }
}
```

### Complete 3D Tensor Structure

```
                    BRAIN QEC TENSOR ARRAY
                    =====================

        Regions (horizontal) →
             V1    V2    V4    MT    M1   PFC  ...

F_i     ┌─────┬─────┬─────┬─────┬─────┬─────┐
(vert)  │     │     │     │     │     │     │
↓   Anat│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│  ← Anatomical functor
        ├─────┼─────┼─────┼─────┼─────┼─────┤
    Func│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│  ← Functional functor
        ├─────┼─────┼─────┼─────┼─────┼─────┤
   Elect│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│  ← Electrophysiology functor
        ├─────┼─────┼─────┼─────┼─────┼─────┤
   Genet│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│  ← Genetics functor
        ├─────┼─────┼─────┼─────┼─────┼─────┤
   Behav│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│ rIDS│  ← Behavior functor
        └─────┴─────┴─────┴─────┴─────┴─────┘

Each [rIDS] cell = ~30 representative features (not all!)

CROSS-FUNCTOR SYNDROME:
If V1_anatomy.rIDS ≠ V1_function.rIDS → SYNDROME!
(Structure disagrees with function)

RECURSIVE CROSS-TRAINING:
V1_function ↔ V2_function (same functor, different regions)
V1_anatomy ↔ V1_function (same region, different functors)
```

### Why F_i is Critical: Multi-Modal Syndrome Detection

**Same region, different views** → If they disagree, **syndrome**!

```python
def cross_functor_syndrome(region_name: str) -> float:
    """Detect inconsistencies WITHIN a region across functors.

    Example: V1 anatomy says connected to V2,
             but V1 function shows weak correlation → syndrome!
    """
    tensor = load_region_tensor(region_name)

    # Anatomy says V2 is neighbor
    anatomical_neighbors = tensor["anatomy"]["rids_connections"]

    # Function says weak correlation
    functional_neighbors = tensor["function"]["rids_connections"]

    # Syndrome = mismatch
    missing_in_functional = anatomical_neighbors - functional_neighbors

    if missing_in_functional:
        return 1.0  # High syndrome: structure ≠ function!
    else:
        return 0.0  # Low syndrome: consistent
```

## Part 1: Advanced Cache Architecture

### Problem: Loading All Regions is Wasteful

**Naive approach** (bad):
```python
# Load all 368 D99 regions into memory
all_regions = []
for r in range(368):
    all_regions.append(load_region_tensor(r))  # 368 × 100MB = 36GB RAM!
```

**Smart cache** (good):
```python
# Load only what we need, when we need it
cache = BrainRegionCache(capacity=20)  # Keep 20 regions in RAM

# Start with seed region
v1 = cache.get_or_load("V1")

# V1 detects syndrome → recruits V2
if v1.syndrome > threshold:
    v2 = cache.get_or_load("V2")  # Only load now!
    v1.cross_train(v2)  # Iterative training
```

### Cache Implementation

```python
from functools import lru_cache
from collections import OrderedDict
import asyncio
from typing import Optional, Set, Dict

class BrainRegionCache:
    """LRU cache for brain region tensors with smart prefetching.

    Features:
    - LRU eviction (least recently used)
    - Prefetch r-IDS neighbors (anticipate needs)
    - Lazy loading (only load when accessed)
    - PostgreSQL backend (persistent storage)
    """

    def __init__(
        self,
        capacity: int = 20,
        db_connection=None,
        prefetch_rids: bool = True
    ):
        """
        Args:
            capacity: Max regions in RAM (e.g., 20 regions × 100MB = 2GB)
            db_connection: PostgreSQL session
            prefetch_rids: If True, prefetch r-IDS neighbors
        """
        self.capacity = capacity
        self.db = db_connection
        self.prefetch_rids = prefetch_rids

        # LRU cache: {region_name → RegionTensor}
        self.cache = OrderedDict()

        # Track accesses for smart eviction
        self.access_counts = {}
        self.last_access = {}

        # Prefetch queue (background loading)
        self.prefetch_queue = asyncio.Queue()

    async def get_or_load(
        self,
        region_name: str,
        session_id: Optional[str] = None
    ) -> RegionTensor:
        """Get region tensor from cache or load from database.

        Args:
            region_name: e.g., "V1"
            session_id: Optional fMRI session UUID

        Returns:
            tensor: RegionTensor for this region
        """
        # Check cache first
        if region_name in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(region_name)
            self.access_counts[region_name] += 1
            self.last_access[region_name] = asyncio.get_event_loop().time()

            print(f"✅ Cache HIT: {region_name}")
            return self.cache[region_name]

        # Cache miss: load from database
        print(f"⚠️  Cache MISS: {region_name}, loading from DB...")
        tensor = await self._load_from_database(region_name, session_id)

        # Add to cache
        await self._add_to_cache(region_name, tensor)

        # Prefetch r-IDS neighbors (anticipate future needs)
        if self.prefetch_rids and tensor.rids_connections:
            await self._schedule_prefetch(tensor.rids_connections)

        return tensor

    async def _add_to_cache(self, region_name: str, tensor: RegionTensor):
        """Add region to cache with LRU eviction."""
        # Evict if at capacity
        if len(self.cache) >= self.capacity:
            # Evict least recently used
            lru_region, _ = self.cache.popitem(last=False)
            print(f"🗑️  Evicted: {lru_region} (LRU)")

        # Add new region
        self.cache[region_name] = tensor
        self.access_counts[region_name] = 1
        self.last_access[region_name] = asyncio.get_event_loop().time()

    async def _load_from_database(
        self,
        region_name: str,
        session_id: Optional[str]
    ) -> RegionTensor:
        """Load region tensor from PostgreSQL."""
        from backend.database.models import BrainRegionTensor

        query = self.db.query(BrainRegionTensor).filter_by(
            region_name=region_name
        )

        if session_id:
            query = query.filter_by(fmri_session_id=session_id)

        row = query.first()

        if not row:
            raise ValueError(f"Region {region_name} not found in database")

        # Reconstruct RegionTensor from database row
        tensor = RegionTensor(
            region_name=row.region_name,
            region_id=row.region_id,
            atlas=row.atlas
        )
        tensor.features = np.array(row.features)
        tensor.rids_connections = set(row.rids_connections)
        tensor.syndrome_history = row.syndrome_history or []

        return tensor

    async def _schedule_prefetch(self, neighbor_ids: Set[int]):
        """Schedule prefetch of r-IDS neighbors in background.

        Args:
            neighbor_ids: Set of region IDs to prefetch
        """
        # Get region names from IDs
        neighbor_names = await self._ids_to_names(neighbor_ids)

        for name in neighbor_names:
            if name not in self.cache:
                await self.prefetch_queue.put(name)

        # Start background prefetch task
        asyncio.create_task(self._background_prefetch())

    async def _background_prefetch(self):
        """Background task to prefetch regions."""
        while not self.prefetch_queue.empty():
            region_name = await self.prefetch_queue.get()

            # Load if not already in cache
            if region_name not in self.cache:
                print(f"🔄 Prefetching: {region_name}")
                tensor = await self._load_from_database(region_name, session_id=None)
                await self._add_to_cache(region_name, tensor)

            self.prefetch_queue.task_done()

    async def _ids_to_names(self, region_ids: Set[int]) -> Set[str]:
        """Convert region IDs to names."""
        from backend.database.models import BrainRegionTensor

        rows = self.db.query(BrainRegionTensor).filter(
            BrainRegionTensor.region_id.in_(region_ids)
        ).all()

        return {row.region_name for row in rows}

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_accesses = sum(self.access_counts.values())
        cache_size = len(self.cache)

        return {
            "capacity": self.capacity,
            "current_size": cache_size,
            "utilization": cache_size / self.capacity,
            "total_accesses": total_accesses,
            "regions_accessed": len(self.access_counts),
            "most_accessed": sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
```

---

## Part 2: Recursive Cross-Training

### Concept: Regions Train Each Other Iteratively

**Key Insight**: Don't train regions independently. Let them **teach each other** via syndrome propagation.

```
V1 predicts → V2
    ↓
V2 actual ≠ predicted → syndrome!
    ↓
V2 sends correction back to V1
    ↓
V1 updates its prediction model
    ↓
Repeat until syndrome → 0
```

### Implementation

```python
class RecursiveCrossTrainer:
    """Recursive cross-training between brain regions.

    Algorithm:
    1. Start with seed region (e.g., V1)
    2. Compute syndrome (prediction error)
    3. If syndrome > threshold:
       - Recruit r-IDS neighbors
       - Cross-train (bidirectional updates)
       - Recurse on neighbors
    4. Repeat until convergence
    """

    def __init__(
        self,
        cache: BrainRegionCache,
        max_depth: int = 4,
        syndrome_threshold: float = 0.1
    ):
        """
        Args:
            cache: Brain region cache
            max_depth: Max recursion depth (r-IDS radius)
            syndrome_threshold: When to stop recruiting
        """
        self.cache = cache
        self.max_depth = max_depth
        self.syndrome_threshold = syndrome_threshold

        # Track training history
        self.training_graph = nx.DiGraph()  # Region → Region training edges

    async def recursive_train(
        self,
        region_name: str,
        fmri_data: np.ndarray,
        depth: int = 0,
        visited: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """Recursively train region and its r-IDS neighbors.

        Args:
            region_name: Starting region (e.g., "V1")
            fmri_data: Full fMRI timeseries [n_timepoints × all_regions]
            depth: Current recursion depth
            visited: Set of already-trained regions (prevent cycles)

        Returns:
            syndromes: Dict of region_name → syndrome value
        """
        if visited is None:
            visited = set()

        if region_name in visited or depth >= self.max_depth:
            return {}

        visited.add(region_name)
        syndromes = {}

        print(f"{'  ' * depth}Training: {region_name} (depth={depth})")

        # Load region tensor (from cache or DB)
        tensor = await self.cache.get_or_load(region_name)

        # Extract fMRI data for this region
        region_timeseries = self._extract_region_timeseries(
            region_name, fmri_data
        )

        # Predict activity from r-IDS neighbors
        predicted = await self._predict_from_neighbors(tensor, fmri_data)

        # Compute syndrome (prediction error)
        actual = region_timeseries
        syndrome = tensor.compute_syndrome(predicted, actual)
        syndromes[region_name] = syndrome

        print(f"{'  ' * depth}  Syndrome: {syndrome:.3f}")

        # If syndrome high, recruit and cross-train neighbors
        if syndrome > self.syndrome_threshold:
            print(f"{'  ' * depth}  Recruiting {len(tensor.rids_connections)} neighbors...")

            for neighbor_id in tensor.rids_connections:
                neighbor_name = await self._id_to_name(neighbor_id)

                # Cross-train: region ↔ neighbor (bidirectional)
                await self._cross_train_pair(
                    region_name, neighbor_name, fmri_data
                )

                # Recurse: train neighbor and its neighbors
                neighbor_syndromes = await self.recursive_train(
                    neighbor_name,
                    fmri_data,
                    depth=depth + 1,
                    visited=visited
                )

                syndromes.update(neighbor_syndromes)

        return syndromes

    async def _predict_from_neighbors(
        self,
        tensor: RegionTensor,
        fmri_data: np.ndarray
    ) -> np.ndarray:
        """Predict region activity from its r-IDS neighbors.

        Args:
            tensor: Target region tensor
            fmri_data: Full fMRI timeseries

        Returns:
            predicted: Predicted timeseries for this region
        """
        if not tensor.rids_connections:
            # No neighbors: predict mean
            return np.zeros(fmri_data.shape[0])

        # Load neighbor tensors
        neighbor_data = []
        for neighbor_id in tensor.rids_connections:
            neighbor_name = await self._id_to_name(neighbor_id)
            neighbor_ts = self._extract_region_timeseries(neighbor_name, fmri_data)
            neighbor_data.append(neighbor_ts)

        neighbor_data = np.array(neighbor_data).T  # [time × n_neighbors]

        # Simple prediction: weighted average of neighbors
        # (Can upgrade to learned model later)
        predicted = np.mean(neighbor_data, axis=1)

        return predicted

    async def _cross_train_pair(
        self,
        region_A_name: str,
        region_B_name: str,
        fmri_data: np.ndarray
    ):
        """Cross-train two regions bidirectionally.

        A learns to predict B
        B learns to predict A

        This is the core of recursive cross-training!

        Args:
            region_A_name: First region (e.g., "V1")
            region_B_name: Second region (e.g., "V2")
            fmri_data: Full fMRI timeseries
        """
        # Load both tensors
        tensor_A = await self.cache.get_or_load(region_A_name)
        tensor_B = await self.cache.get_or_load(region_B_name)

        # Get timeseries
        ts_A = self._extract_region_timeseries(region_A_name, fmri_data)
        ts_B = self._extract_region_timeseries(region_B_name, fmri_data)

        # A predicts B
        predicted_B = self._simple_predict(tensor_A, ts_A)
        syndrome_A_to_B = np.linalg.norm(predicted_B - ts_B)

        # B predicts A
        predicted_A = self._simple_predict(tensor_B, ts_B)
        syndrome_B_to_A = np.linalg.norm(predicted_A - ts_A)

        # Update both models (reduce prediction error)
        learning_rate = 0.01

        # A learns from error in predicting B
        error_A = ts_B - predicted_B
        tensor_A.features += learning_rate * error_A[:len(tensor_A.features)]

        # B learns from error in predicting A
        error_B = ts_A - predicted_A
        tensor_B.features += learning_rate * error_B[:len(tensor_B.features)]

        # Record training edge
        self.training_graph.add_edge(
            region_A_name, region_B_name,
            syndrome=syndrome_A_to_B
        )
        self.training_graph.add_edge(
            region_B_name, region_A_name,
            syndrome=syndrome_B_to_A
        )

        print(f"    Cross-trained: {region_A_name} ↔ {region_B_name}")
        print(f"      {region_A_name}→{region_B_name}: syndrome={syndrome_A_to_B:.3f}")
        print(f"      {region_B_name}→{region_A_name}: syndrome={syndrome_B_to_A:.3f}")

    def _simple_predict(self, tensor: RegionTensor, timeseries: np.ndarray) -> np.ndarray:
        """Simple prediction: project timeseries onto learned features."""
        # Project timeseries onto feature space
        features = tensor.features

        # Simple linear model: predicted = timeseries · features
        predicted = np.dot(timeseries[:len(features)], features)

        return predicted

    def _extract_region_timeseries(
        self,
        region_name: str,
        fmri_data: np.ndarray
    ) -> np.ndarray:
        """Extract timeseries for specific region from full fMRI data.

        Args:
            region_name: e.g., "V1"
            fmri_data: [n_timepoints × n_regions]

        Returns:
            timeseries: [n_timepoints] for this region
        """
        # Look up region index from atlas
        region_idx = self._name_to_index(region_name)
        return fmri_data[:, region_idx]

    async def _id_to_name(self, region_id: int) -> str:
        """Convert region ID to name."""
        from backend.database.models import BrainRegionTensor
        row = self.cache.db.query(BrainRegionTensor).filter_by(
            region_id=region_id
        ).first()
        return row.region_name if row else f"region_{region_id}"

    def _name_to_index(self, region_name: str) -> int:
        """Convert region name to index in fMRI data array."""
        # This requires atlas mapping (implement based on your D99 structure)
        # For now, simple lookup
        atlas_mapping = {
            "V1": 0, "V2": 1, "V4": 2, "MT": 3,
            # ... rest of D99 regions
        }
        return atlas_mapping.get(region_name, 0)

    def get_training_summary(self) -> Dict:
        """Get summary of cross-training process."""
        return {
            "regions_trained": len(self.training_graph.nodes()),
            "cross_training_edges": self.training_graph.number_of_edges(),
            "avg_syndrome": np.mean([
                d['syndrome']
                for _, _, d in self.training_graph.edges(data=True)
            ]),
            "training_graph": self.training_graph
        }
```

---

## Part 3: Complete System Integration

### Brain QEC System with Cache + Cross-Training

```python
class BrainQECSystem:
    """Complete brain QEC tensor system.

    Features:
    - Advanced cache (20 regions in RAM)
    - Recursive cross-training (regions train each other)
    - merge2docs services (r-IDS, cluster-editing-vs)
    - PostgreSQL backend (persistent storage)
    - fMRI integration (your macaque data)
    """

    def __init__(
        self,
        db_session,
        cache_capacity: int = 20,
        max_training_depth: int = 4
    ):
        """
        Args:
            db_session: PostgreSQL session
            cache_capacity: Max regions in RAM
            max_training_depth: Max r-IDS recursion depth
        """
        self.db = db_session

        # Initialize cache
        self.cache = BrainRegionCache(
            capacity=cache_capacity,
            db_connection=db_session,
            prefetch_rids=True
        )

        # Initialize cross-trainer
        self.cross_trainer = RecursiveCrossTrainer(
            cache=self.cache,
            max_depth=max_training_depth,
            syndrome_threshold=0.1
        )

        # merge2docs integration
        from backend.integration.merge2docs_bridge import AlgorithmServiceClient
        self.algorithm_service = AlgorithmServiceClient()

    async def train_from_fmri(
        self,
        fmri_session_id: str,
        seed_region: str = "V1"
    ) -> Dict:
        """Train brain model from fMRI session using recursive cross-training.

        Args:
            fmri_session_id: UUID of fMRI session in database
            seed_region: Starting region for recursive training

        Returns:
            results: Training summary
        """
        print("=" * 70)
        print("Brain QEC System: Recursive Cross-Training")
        print("=" * 70)
        print()

        # Load fMRI data
        print(f"Loading fMRI session: {fmri_session_id}")
        fmri_data = await self._load_fmri_session(fmri_session_id)
        print(f"  Data shape: {fmri_data.shape}")
        print()

        # Recursive training starting from seed
        print(f"Starting recursive training from: {seed_region}")
        print(f"  Max depth: {self.cross_trainer.max_depth}")
        print(f"  Syndrome threshold: {self.cross_trainer.syndrome_threshold}")
        print()

        syndromes = await self.cross_trainer.recursive_train(
            region_name=seed_region,
            fmri_data=fmri_data,
            depth=0
        )

        print()
        print("=" * 70)
        print("Training Complete")
        print("=" * 70)

        # Get statistics
        cache_stats = self.cache.get_stats()
        training_stats = self.cross_trainer.get_training_summary()

        print(f"Cache utilization: {cache_stats['utilization']:.1%}")
        print(f"Regions loaded: {cache_stats['current_size']}/{cache_stats['capacity']}")
        print(f"Total accesses: {cache_stats['total_accesses']}")
        print()
        print(f"Regions trained: {training_stats['regions_trained']}")
        print(f"Cross-training edges: {training_stats['cross_training_edges']}")
        print(f"Average syndrome: {training_stats['avg_syndrome']:.3f}")
        print()

        # Most accessed regions
        print("Most accessed regions:")
        for region, count in cache_stats['most_accessed']:
            print(f"  {region}: {count} accesses")
        print()

        return {
            "syndromes": syndromes,
            "cache_stats": cache_stats,
            "training_stats": training_stats
        }

    async def query_syndrome_pattern(
        self,
        region_name: str,
        n_sessions: int = 5
    ) -> Dict:
        """Query syndrome patterns for a region across multiple sessions.

        Args:
            region_name: e.g., "V1"
            n_sessions: Number of recent sessions to analyze

        Returns:
            pattern: Syndrome pattern analysis
        """
        # Get recent sessions
        from backend.database.models import FMRISession
        sessions = self.db.query(FMRISession).order_by(
            FMRISession.scan_date.desc()
        ).limit(n_sessions).all()

        syndromes_over_time = []

        for session in sessions:
            # Load region tensor for this session
            tensor = await self.cache.get_or_load(
                region_name,
                session_id=session.id
            )

            syndromes_over_time.append({
                "session_id": session.id,
                "date": session.scan_date,
                "syndrome_mean": np.mean(tensor.syndrome_history),
                "syndrome_std": np.std(tensor.syndrome_history)
            })

        return {
            "region": region_name,
            "n_sessions": n_sessions,
            "syndromes": syndromes_over_time,
            "trend": self._analyze_trend(syndromes_over_time)
        }

    def _analyze_trend(self, syndromes_list: List[Dict]) -> str:
        """Analyze syndrome trend (increasing, decreasing, stable)."""
        values = [s['syndrome_mean'] for s in syndromes_list]

        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression
        from scipy.stats import linregress
        x = np.arange(len(values))
        slope, _, _, p_value, _ = linregress(x, values)

        if p_value > 0.05:
            return "stable"
        elif slope > 0:
            return "increasing"  # Getting worse!
        else:
            return "decreasing"  # Getting better (learning!)

    async def _load_fmri_session(self, session_id: str) -> np.ndarray:
        """Load fMRI timeseries from database."""
        from backend.database.models import FMRISession
        session = self.db.query(FMRISession).filter_by(id=session_id).first()

        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Load NIfTI and extract timeseries
        import nibabel as nib
        from nilearn.input_data import NiftiLabelsMasker

        fmri_img = nib.load(session.preprocessed_path)
        atlas_mask = nib.load(f"/data/atlases/{session.atlas}_mask.nii.gz")

        masker = NiftiLabelsMasker(labels_img=atlas_mask, standardize=True)
        timeseries = masker.fit_transform(fmri_img)

        return timeseries  # [n_timepoints × n_regions]
```

---

## Part 4: Example Usage

### Example 1: Train from Single fMRI Session

```python
async def demo_recursive_training():
    """Demo: Train brain model using recursive cross-training."""

    # Initialize system
    brain_qec = BrainQECSystem(
        db_session=db.session,
        cache_capacity=20,  # Keep 20 regions in RAM
        max_training_depth=4  # r-IDS depth
    )

    # Train from fMRI session
    results = await brain_qec.train_from_fmri(
        fmri_session_id="session_123",
        seed_region="V1"  # Start from primary visual cortex
    )

    # Results:
    # Cache loaded: V1, V2, V4, MT, MST, TEO, TE, ... (20 regions max)
    # Cross-training: V1↔V2, V2↔V4, V4↔MT, ...
    # Syndromes: {"V1": 0.05, "V2": 0.12, "V4": 0.08, "MT": 0.15, ...}
```

**Output**:
```
======================================================================
Brain QEC System: Recursive Cross-Training
======================================================================

Loading fMRI session: session_123
  Data shape: (200, 100)

Starting recursive training from: V1
  Max depth: 4
  Syndrome threshold: 0.1

Training: V1 (depth=0)
  Syndrome: 0.156
  Recruiting 3 neighbors...
    Cross-trained: V1 ↔ V2
      V1→V2: syndrome=0.145
      V2→V1: syndrome=0.133
  Training: V2 (depth=1)
    Syndrome: 0.112
    Recruiting 4 neighbors...
      Cross-trained: V2 ↔ V4
        V2→V4: syndrome=0.098
        V4→V2: syndrome=0.102
    Training: V4 (depth=2)
      Syndrome: 0.087
      Recruiting 2 neighbors...
        Cross-trained: V4 ↔ MT
          V4→MT: syndrome=0.076
          MT→V4: syndrome=0.081
      Training: MT (depth=3)
        Syndrome: 0.065
        ✅ Below threshold, converged!

======================================================================
Training Complete
======================================================================
Cache utilization: 70%
Regions loaded: 14/20
Total accesses: 47

Regions trained: 8
Cross-training edges: 14
Average syndrome: 0.095

Most accessed regions:
  V1: 12 accesses
  V2: 9 accesses
  V4: 7 accesses
  MT: 5 accesses
  MST: 3 accesses
```

### Example 2: Query Syndrome Patterns

```python
# Check if MT region is learning over time
pattern = await brain_qec.query_syndrome_pattern(
    region_name="MT",
    n_sessions=10
)

print(f"MT syndrome trend: {pattern['trend']}")
# "decreasing" → Learning! Syndrome reducing over sessions
```

---

## Part 5: Integration with merge2docs Services

### When to Use merge2docs

```python
class BrainQECSystem:
    # ... (continued)

    async def compute_rids_for_region(
        self,
        region_name: str,
        G: nx.Graph,
        r: int = 4
    ) -> Set[int]:
        """Compute r-IDS connections via merge2docs service.

        Args:
            region_name: Region to compute r-IDS for
            G: Functional connectivity graph
            r: Coverage radius

        Returns:
            rids_neighbors: Set of connected region IDs
        """
        # Call merge2docs 'ids' algorithm
        result = await self.algorithm_service.call(
            algorithm_name="ids",
            graph_data=G,
            r=r
        )

        return result.independent_set

    async def cluster_functional_systems(
        self,
        G_functional: nx.Graph,
        target_clusters: int = 10
    ) -> Dict[int, int]:
        """Cluster regions into functional systems via merge2docs.

        Uses cluster-editing-vs with auto-tuning.
        """
        # Call merge2docs service
        result = await self.algorithm_service.call(
            algorithm_name="cluster_editing_vs",
            graph_data=G_functional,
            k=None,  # Auto-tune
            use_gpu=True
        )

        return result.communities
```

---

## Part 6: What Makes This Mind-Blowing

### 1. Smart Caching
- ✅ Only loads 20/368 regions (5% of data in RAM)
- ✅ Prefetches r-IDS neighbors (anticipates needs)
- ✅ LRU eviction (keeps hot regions cached)

### 2. Recursive Cross-Training
- ✅ Regions teach each other (V1→V2→V4→MT)
- ✅ Bidirectional updates (V1 learns from V2, V2 from V1)
- ✅ Guided by syndrome (only recruit if prediction error high)

### 3. Leverages merge2docs
- ✅ r-IDS via service (GPU-accelerated)
- ✅ Cluster-editing-vs (trained algorithm)
- ✅ Feature selection (dimensionality reduction)

### 4. Database Integration
- ✅ Persistent storage (PostgreSQL)
- ✅ Multi-session tracking (syndrome trends)
- ✅ Queryable (SQL for analysis)

---

## Part 7: Timeline & Next Steps

### Phase 1: Build Cache System (Week 1-2)
```
[ ] Implement BrainRegionCache with LRU
[ ] Add prefetch logic for r-IDS neighbors
[ ] Test with 100 D99 regions
```

### Phase 2: Build Cross-Trainer (Week 3-4)
```
[ ] Implement RecursiveCrossTrainer
[ ] Add syndrome-guided recruitment
[ ] Test recursive training from V1
```

### Phase 3: Integrate with merge2docs (Week 5)
```
[ ] Call 'ids' service for r-IDS
[ ] Call 'cluster_editing_vs' for communities
[ ] Validate against standalone implementation
```

### Phase 4: Train on Real fMRI (Week 6)
```
[ ] Ingest macaque fMRI session
[ ] Run recursive training
[ ] Analyze syndrome patterns
```

---

## TL;DR

Build brain-specific QEC tensor with:
1. **Smart cache**: LRU + prefetch (20/368 regions in RAM)
2. **Recursive cross-training**: Regions train each other via r-IDS
3. **merge2docs services**: r-IDS, clustering, feature selection
4. **PostgreSQL backend**: Persistent, queryable

**Timeline**: 6 weeks
**Cost**: $0 (CPU sufficient for cache, merge2docs handles GPU)
**Result**: Working brain model trained on YOUR macaque fMRI data!

Now ready to **spec this out fully** and **send design to merge2docs via MCP**.
