# Practical Brain QEC Tensor: One Column Per Region

## Executive Summary

**Simple, immediate approach**: Build **one QEC tensor column per brain region** using existing fMRI data.

```
Region-Specific Tensors (100 regions from D99):

V1_tensor → learn visual features for primary visual cortex
MT_tensor → learn motion features for middle temporal area
M1_tensor → learn motor features for primary motor cortex
...
PFC_tensor → learn executive features for prefrontal cortex
```

Each tensor learns:
- **What features** this region represents
- **How it responds** to stimuli (fMRI timeseries)
- **Which other regions** it communicates with (r-IDS connections)
- **Error patterns** (syndrome detection for this region)

## Why This Works

### You Already Have:
- ✅ **fMRI data** (macaque 4D timeseries)
- ✅ **D99 atlas** (100 cortical regions defined)
- ✅ **merge2docs services** (cluster-editing-vs, r-IDS)
- ✅ **Database** (to store learned features per region)

### What You Build:
- **One tensor column per region** (modular, incremental)
- **Simple at first** (just fMRI features)
- **Expand later** (add genetics, anatomy, etc.)

---

## Architecture: Region-Specific Tensor

### Structure for Each Region

```python
class RegionTensor:
    """QEC tensor for a single brain region.

    Example: V1_tensor for primary visual cortex.

    Stores:
    - Features learned from fMRI timeseries
    - Connections to other regions (via r-IDS)
    - Syndrome patterns (prediction errors)
    - Learned corrections
    """

    def __init__(self, region_name: str, region_id: int, atlas="D99"):
        self.region_name = region_name  # "V1"
        self.region_id = region_id  # 42 (from D99)
        self.atlas = atlas

        # Learned representations
        self.features = None  # [n_features] vector
        self.r_ids_hubs = None  # Set of connected regions (r=4)

        # QEC components
        self.syndrome_history = []  # Track prediction errors
        self.correction_history = []  # Track how we fixed them

        # Database storage
        self.db_id = None  # UUID in PostgreSQL

    def extract_features_from_fmri(
        self,
        timeseries: np.ndarray,  # [n_timepoints] for this region
        stimuli: np.ndarray  # [n_timepoints × stimulus_dims]
    ) -> np.ndarray:
        """Learn what this region encodes.

        For V1: Edge orientation, spatial frequency
        For MT: Motion direction, speed
        For PFC: Task rules, working memory

        Args:
            timeseries: BOLD signal for this region
            stimuli: What the monkey saw/did

        Returns:
            features: [n_features] learned representation
        """
        # Option 1: Simple (PCA)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        features = pca.fit_transform(timeseries.reshape(-1, 1))

        # Option 2: Better (diffusion CNN encoder)
        # features = self.diffusion_encoder(timeseries, stimuli)

        self.features = features
        return features

    def compute_syndrome(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """Measure prediction error for this region.

        Args:
            predicted: What we predicted this region would do
            actual: What it actually did (fMRI measurement)

        Returns:
            syndrome: Prediction error magnitude
        """
        syndrome = np.linalg.norm(predicted - actual)
        self.syndrome_history.append(syndrome)
        return syndrome

    async def retrieve_connected_regions(
        self,
        G_functional: nx.Graph,
        r: int = 4
    ) -> Set[int]:
        """Find which other regions this one talks to (r-IDS).

        Args:
            G_functional: Functional connectivity graph
            r: Coverage radius (4 = multi-synaptic)

        Returns:
            connected_regions: Set of region IDs within r hops
        """
        # Use merge2docs service for r-IDS
        from backend.integration.merge2docs_bridge import call_algorithm_service

        result = await call_algorithm_service(
            algorithm_name="ids",
            graph_data=G_functional,
            start_node=self.region_id,
            r=r
        )

        self.r_ids_hubs = result.independent_set
        return self.r_ids_hubs

    def save_to_database(self, db_session):
        """Persist this region's tensor to PostgreSQL."""
        from backend.database.models import BrainRegionTensor

        tensor_row = BrainRegionTensor(
            region_name=self.region_name,
            region_id=self.region_id,
            atlas=self.atlas,
            features=self.features.tolist(),  # JSON
            rids_connections=list(self.r_ids_hubs),
            syndrome_mean=np.mean(self.syndrome_history),
            syndrome_std=np.std(self.syndrome_history)
        )

        db_session.add(tensor_row)
        db_session.commit()

        self.db_id = tensor_row.id
```

---

## Database Schema

### Table: `brain_region_tensors`

```sql
CREATE TABLE brain_region_tensors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),

    -- Region identity
    region_name TEXT NOT NULL,  -- 'V1', 'MT', 'M1', etc.
    region_id INTEGER NOT NULL,  -- From D99 atlas
    atlas TEXT DEFAULT 'D99',
    hemisphere TEXT,  -- 'left', 'right', 'bilateral'

    -- Learned features
    features JSONB,  -- [n_features] vector as JSON
    feature_dim INTEGER,  -- Dimension of feature vector
    feature_method TEXT,  -- 'pca', 'diffusion_cnn', 'ica'

    -- Connections (r-IDS)
    rids_connections INTEGER[],  -- Array of region IDs
    rids_radius INTEGER DEFAULT 4,

    -- QEC syndrome
    syndrome_mean REAL,
    syndrome_std REAL,
    syndrome_history JSONB,  -- Time series of syndromes

    -- Metadata
    fmri_session_id UUID REFERENCES fmri_sessions(id),
    n_timepoints INTEGER,
    stimuli_type TEXT,  -- 'visual', 'motor', 'cognitive'

    UNIQUE(region_id, atlas, fmri_session_id)
);

CREATE INDEX idx_region_tensors_name ON brain_region_tensors(region_name);
CREATE INDEX idx_region_tensors_syndrome ON brain_region_tensors(syndrome_mean);
CREATE INDEX idx_region_tensors_session ON brain_region_tensors(fmri_session_id);
```

### Table: `fmri_sessions`

```sql
CREATE TABLE fmri_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),

    -- Subject
    subject_id TEXT NOT NULL,  -- 'macaque_01'
    species TEXT DEFAULT 'macaque',
    sex TEXT,
    age_months INTEGER,

    -- Scan parameters
    scan_date DATE,
    tr REAL,  -- Repetition time (seconds)
    n_volumes INTEGER,  -- Number of timepoints
    voxel_size_mm REAL,

    -- Task
    task_name TEXT,  -- 'rest', 'visual_motion', 'reach_grasp'
    stimuli_file TEXT,  -- Path to stimuli log

    -- Files
    nifti_path TEXT,  -- Path to 4D NIfTI
    preprocessed_path TEXT,  -- Path after preprocessing

    UNIQUE(subject_id, scan_date, task_name)
);
```

---

## Pipeline: Build Tensor for Each Region

### Step 1: Ingest fMRI Session

```python
async def ingest_fmri_session(
    fmri_path: str,
    atlas_mask_path: str,
    task_name: str = "rest"
) -> UUID:
    """Ingest 4D fMRI and extract timeseries per region.

    Args:
        fmri_path: Path to 4D NIfTI file
        atlas_mask_path: D99 atlas mask
        task_name: Experimental task

    Returns:
        session_id: UUID for this session
    """
    import nibabel as nib
    from nilearn.input_data import NiftiLabelsMasker

    # Load fMRI
    fmri_img = nib.load(fmri_path)

    # Extract timeseries per region
    masker = NiftiLabelsMasker(
        labels_img=atlas_mask_path,
        standardize=True,
        detrend=True
    )
    timeseries = masker.fit_transform(fmri_img)  # [n_timepoints × n_regions]

    # Save session to database
    session = FMRISession(
        subject_id="macaque_01",
        species="macaque",
        task_name=task_name,
        nifti_path=fmri_path,
        tr=2.0,
        n_volumes=timeseries.shape[0]
    )
    db.session.add(session)
    db.session.commit()

    return session.id, timeseries
```

### Step 2: Build Tensor for Each Region

```python
async def build_region_tensors(
    session_id: UUID,
    timeseries: np.ndarray,  # [n_timepoints × n_regions]
    atlas_regions: List[Dict]  # From D99 atlas
):
    """Build one tensor per region.

    Args:
        session_id: fMRI session UUID
        timeseries: Extracted timeseries [time × regions]
        atlas_regions: List of region metadata from D99
    """
    n_regions = len(atlas_regions)

    for i, region_meta in enumerate(atlas_regions):
        print(f"Processing {i+1}/{n_regions}: {region_meta['name']}")

        # Create tensor for this region
        region_tensor = RegionTensor(
            region_name=region_meta['name'],
            region_id=region_meta['id'],
            atlas="D99"
        )

        # Extract features from fMRI timeseries for this region
        region_timeseries = timeseries[:, i]  # [n_timepoints]
        features = region_tensor.extract_features_from_fmri(
            region_timeseries,
            stimuli=None  # Or load from task
        )

        print(f"  Features: {features.shape}")

        # Save to database
        region_tensor.save_to_database(db.session)

    print(f"✅ Built {n_regions} region tensors")
```

### Step 3: Compute r-IDS Connections

```python
async def compute_region_connections(session_id: UUID):
    """Compute r-IDS connections between regions.

    Uses functional connectivity from fMRI.
    """
    # Load all region tensors for this session
    region_tensors = db.session.query(BrainRegionTensor).filter_by(
        fmri_session_id=session_id
    ).all()

    # Build functional connectivity graph
    features_matrix = np.array([r.features for r in region_tensors])  # [n_regions × n_features]
    from sklearn.metrics.pairwise import cosine_similarity
    connectivity = cosine_similarity(features_matrix)

    # Build graph
    G_functional = nx.Graph()
    for i, r in enumerate(region_tensors):
        G_functional.add_node(r.region_id, name=r.region_name)

    threshold = 0.3
    for i in range(len(region_tensors)):
        for j in range(i+1, len(region_tensors)):
            if connectivity[i, j] > threshold:
                G_functional.add_edge(
                    region_tensors[i].region_id,
                    region_tensors[j].region_id,
                    weight=connectivity[i, j]
                )

    print(f"Functional graph: {G_functional.number_of_nodes()} nodes, {G_functional.number_of_edges()} edges")

    # Compute r-IDS for each region
    for region_tensor in region_tensors:
        rt = RegionTensor(
            region_name=region_tensor.region_name,
            region_id=region_tensor.region_id
        )

        # Find connected regions via r-IDS
        connected = await rt.retrieve_connected_regions(G_functional, r=4)

        # Update database
        region_tensor.rids_connections = list(connected)
        db.session.commit()

        print(f"  {region_tensor.region_name}: {len(connected)} r-IDS connections")

    print("✅ Computed r-IDS connections for all regions")
```

### Step 4: Syndrome Detection Across Regions

```python
def compute_cross_region_syndromes(session_id: UUID):
    """Detect inconsistencies between regions.

    Example: V1 and MT should have correlated motion responses.
    If V1 responds to motion but MT doesn't → syndrome!
    """
    region_tensors = db.session.query(BrainRegionTensor).filter_by(
        fmri_session_id=session_id
    ).all()

    syndromes = {}

    # Check known anatomical pathways
    known_pathways = [
        ("V1", "V2"),  # Visual hierarchy
        ("V2", "V4"),
        ("V4", "MT"),
        ("M1", "SMA"),  # Motor hierarchy
        ("PFC", "ACC"),  # Executive
    ]

    for source_name, target_name in known_pathways:
        # Find tensors
        source = next((r for r in region_tensors if r.region_name == source_name), None)
        target = next((r for r in region_tensors if r.region_name == target_name), None)

        if source is None or target is None:
            continue

        # Check if they're connected in r-IDS
        if target.region_id not in source.rids_connections:
            # Anatomical pathway exists, but functional connection missing → syndrome!
            syndrome = 1.0
            syndromes[f"{source_name}_to_{target_name}"] = syndrome
            print(f"⚠️  Syndrome: {source_name} → {target_name} pathway disconnected")

    return syndromes
```

---

## Example: Complete Pipeline

```python
async def main():
    """Complete pipeline: fMRI → region tensors → connections → syndromes."""

    print("=" * 70)
    print("Building Region-Specific QEC Tensors")
    print("=" * 70)
    print()

    # Step 1: Ingest fMRI session
    print("Step 1: Ingesting fMRI session...")
    session_id, timeseries = await ingest_fmri_session(
        fmri_path="/data/macaque_01_rest_fmri.nii.gz",
        atlas_mask_path="/data/atlases/D99_atlas_mask.nii.gz",
        task_name="rest"
    )
    print(f"  Session ID: {session_id}")
    print(f"  Timeseries: {timeseries.shape}")
    print()

    # Step 2: Get regions from D99 atlas
    print("Step 2: Querying D99 atlas...")
    atlas_client = BrainAtlasClient("http://localhost:8007")
    regions = atlas_client.list_regions(species="macaque", atlas="D99", limit=100)
    print(f"  Retrieved {len(regions)} regions")
    print()

    # Step 3: Build tensor for each region
    print("Step 3: Building region tensors...")
    await build_region_tensors(session_id, timeseries, regions)
    print()

    # Step 4: Compute connections via r-IDS
    print("Step 4: Computing r-IDS connections...")
    await compute_region_connections(session_id)
    print()

    # Step 5: Detect syndromes
    print("Step 5: Detecting cross-region syndromes...")
    syndromes = compute_cross_region_syndromes(session_id)
    print(f"  Found {len(syndromes)} syndrome patterns")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Session: {session_id}")
    print(f"Regions: {len(regions)} tensors built")
    print(f"Connections: r-IDS with r=4")
    print(f"Syndromes: {len(syndromes)} inconsistencies detected")
    print()
    print("✅ Region-specific QEC tensor array complete!")
```

---

## Query Interface: Use the Tensors

### Query 1: Which regions encode visual motion?

```python
def find_motion_regions(session_id: UUID) -> List[str]:
    """Find regions that respond to motion stimuli."""

    # Get all regions
    regions = db.session.query(BrainRegionTensor).filter_by(
        fmri_session_id=session_id
    ).all()

    # Load motion stimulus
    motion_stimulus = load_motion_stimulus()  # [n_timepoints]

    motion_regions = []

    for region in regions:
        # Correlate region features with motion
        correlation = np.corrcoef(region.features, motion_stimulus)[0, 1]

        if correlation > 0.7:
            motion_regions.append(region.region_name)

    return motion_regions

# Result: ['V1', 'V2', 'V4', 'MT', 'MST']
```

### Query 2: What is the r-IDS backbone for visual system?

```python
def get_visual_backbone(session_id: UUID) -> Set[str]:
    """Get r-IDS backbone for visual regions."""

    visual_system = ['V1', 'V2', 'V3', 'V4', 'MT', 'MST', 'TEO', 'TE']

    # Get visual region tensors
    visual_tensors = db.session.query(BrainRegionTensor).filter(
        BrainRegionTensor.region_name.in_(visual_system),
        BrainRegionTensor.fmri_session_id == session_id
    ).all()

    # Build subgraph
    G_visual = nx.Graph()
    for r in visual_tensors:
        G_visual.add_node(r.region_id, name=r.region_name)
        for connected_id in r.rids_connections:
            if connected_id in [t.region_id for t in visual_tensors]:
                G_visual.add_edge(r.region_id, connected_id)

    # Compute r-IDS backbone
    rids = RadiusIDS(r=4)
    backbone_ids = rids.compute(G_visual)

    # Map back to names
    id_to_name = {r.region_id: r.region_name for r in visual_tensors}
    backbone_names = {id_to_name[rid] for rid in backbone_ids}

    return backbone_names

# Result: {'V1', 'V4', 'MT', 'TE'}  # 4 hubs from 8 regions
```

### Query 3: Syndrome patterns for this subject?

```python
def analyze_subject_syndromes(subject_id: str) -> Dict:
    """Analyze syndrome patterns across all sessions for one subject."""

    # Get all sessions for this subject
    sessions = db.session.query(FMRISession).filter_by(
        subject_id=subject_id
    ).all()

    all_syndromes = []

    for session in sessions:
        syndromes = compute_cross_region_syndromes(session.id)
        all_syndromes.append({
            "session": session.task_name,
            "date": session.scan_date,
            "syndromes": syndromes
        })

    # Find persistent syndromes (appear in >80% of sessions)
    from collections import Counter
    syndrome_counts = Counter()

    for s in all_syndromes:
        for syndrome_key in s["syndromes"].keys():
            syndrome_counts[syndrome_key] += 1

    persistent = {
        k: v for k, v in syndrome_counts.items()
        if v >= 0.8 * len(sessions)
    }

    return {
        "subject": subject_id,
        "n_sessions": len(sessions),
        "persistent_syndromes": persistent,
        "all_syndromes": all_syndromes
    }
```

---

## Benefits of This Approach

### 1. Modular and Incremental
- ✅ Build one region at a time
- ✅ Add more as fMRI data becomes available
- ✅ No need to process all 368 D99 regions at once

### 2. Uses Existing Infrastructure
- ✅ merge2docs services (cluster-editing-vs, r-IDS)
- ✅ D99 atlas (via brain_atlas_http_server)
- ✅ PostgreSQL database (already have)
- ✅ Your fMRI data (the valuable part!)

### 3. Queryable and Analyzable
- ✅ "Which regions respond to motion?"
- ✅ "What's the r-IDS backbone for visual system?"
- ✅ "Are there persistent syndrome patterns?"

### 4. Upgradeable
- Start: PCA features (simple, fast)
- Later: Diffusion CNN features (better)
- Even later: Multi-modal (fMRI + genetics + anatomy)

---

## Timeline

| Week | Task | Output |
|------|------|--------|
| 1 | Implement RegionTensor class | Working code |
| 2 | Create database schema | PostgreSQL tables |
| 3 | Ingest first fMRI session | 100 region tensors |
| 4 | Compute r-IDS connections | Functional graph |
| 5 | Implement syndrome detection | Cross-region validation |
| 6 | Build query interface | Analysis tools |

**Total**: 6 weeks for working system with real data

---

## Cost

| Item | Cost |
|------|------|
| **Compute** (PCA features) | ~$0 (CPU sufficient) |
| **Storage** (PostgreSQL) | ~$0 (local) |
| **Data** (PRIME-DE fMRI) | ~$0 (public) |
| **merge2docs services** | ~$0 (already running) |
| **TOTAL** | **$0** |

If upgrading to diffusion CNN: ~$400 for GPU training (one-time)

---

## TL;DR

**Build ONE tensor column per brain region** using:
- ✅ Your existing fMRI data
- ✅ D99 atlas (100 regions)
- ✅ merge2docs services (r-IDS)
- ✅ PostgreSQL database

**Result**: Queryable brain model with region-specific features, connections, and syndrome detection.

**Timeline**: 6 weeks
**Cost**: $0 (or $400 with diffusion CNN upgrade)

**This is the practical, immediate version!**
