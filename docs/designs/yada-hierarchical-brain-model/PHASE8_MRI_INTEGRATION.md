# Phase 8: Semantic Hierarchy from 4D MRI Activity Patterns

## Overview

This phase builds the **semantic (functional) hierarchy** by extracting activity patterns from 4D macaque MRI data using diffusion CNN models. This complements the syntactic (anatomical) hierarchy from Phase 7.

## Data Sources

### 1. 4D fMRI Time-Series Data

**Macaque Resting-State and Task fMRI:**
- **Format**: NIfTI 4D volumes (x, y, z, time)
- **Temporal resolution**: TR = 1-2 seconds
- **Spatial resolution**: 1-2 mm isotropic
- **Duration**: 5-30 minutes per scan
- **Species**: *Macaca mulatta* (rhesus macaque)
- **Atlas alignment**: D99 stereotaxic space

**Typical datasets:**
- Resting-state: Spontaneous activity patterns
- Visual tasks: Responses to natural images, gratings
- Motor tasks: Reach-to-grasp, saccades
- Cognitive tasks: Working memory, attention

### 2. Diffusion CNN Feature Extraction

**Inspiration**: Diffusion models for neuroimaging (e.g., Benedict et al.)

**Architecture Options:**

#### Option A: Denoising Diffusion Probabilistic Models (DDPM)
```
4D fMRI [T×X×Y×Z] → 3D CNN Encoder → Latent Features [T×D]
                                    ↓
                            Diffusion Process
                                    ↓
                            Denoised Features [T'×D']
                                    ↓
                        Functional Graph Construction
```

#### Option B: Variational Diffusion Models
```
fMRI Timeseries → VAE Encoder → Latent z ~ N(μ,σ²)
                                    ↓
                          Diffusion Dynamics
                                    ↓
                        Functional Connectivity
```

#### Option C: Graph Diffusion CNNs
```
fMRI Regions → Correlation → Functional Graph
                                    ↓
                          Graph Diffusion CNN
                                    ↓
                        Community Detection
```

## Pipeline

### Step 1: Preprocessing

```python
import nibabel as nib
from nilearn import image, masking

def preprocess_fmri(fmri_path: str, atlas_mask: str) -> np.ndarray:
    """Preprocess 4D fMRI data.

    Args:
        fmri_path: Path to 4D NIfTI file
        atlas_mask: D99 atlas mask

    Returns:
        timeseries: [n_regions × n_timepoints] array
    """
    # Load fMRI
    fmri_img = nib.load(fmri_path)

    # Motion correction (using nilearn.image.smooth_img)
    smoothed = image.smooth_img(fmri_img, fwhm=6.0)

    # Extract region timeseries
    from nilearn.input_data import NiftiLabelsMasker
    masker = NiftiLabelsMasker(
        labels_img=atlas_mask,
        standardize=True,
        detrend=True,
        high_pass=0.01,  # Hz
        low_pass=0.1,     # Hz
        t_r=2.0
    )

    timeseries = masker.fit_transform(smoothed)  # [n_timepoints × n_regions]

    return timeseries.T  # [n_regions × n_timepoints]
```

### Step 2: Diffusion CNN Feature Extraction

**Option 1: Use Pre-trained Diffusion Model**

```python
import torch
from diffusers import DDPMPipeline
from torch import nn

class fMRIDiffusionEncoder(nn.Module):
    """Diffusion-based encoder for fMRI features."""

    def __init__(self, n_regions=100, hidden_dim=256, latent_dim=64):
        super().__init__()

        # Temporal CNN
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(n_regions, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Diffusion process
        self.diffusion = DDPMPipeline.from_pretrained(
            "google/ddpm-celebahq-256"  # Placeholder - need neuroimaging model
        )

        # Latent projection
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, timeseries):
        """
        Args:
            timeseries: [batch × n_regions × n_timepoints]

        Returns:
            features: [batch × latent_dim]
        """
        # Extract temporal features
        x = self.temporal_encoder(timeseries)  # [batch × hidden × 1]
        x = x.squeeze(-1)  # [batch × hidden]

        # Project to latent space
        z = self.projection(x)  # [batch × latent_dim]

        return z


def extract_diffusion_features(timeseries: np.ndarray) -> np.ndarray:
    """Extract features using diffusion CNN.

    Args:
        timeseries: [n_regions × n_timepoints]

    Returns:
        features: [n_regions × latent_dim]
    """
    model = fMRIDiffusionEncoder(
        n_regions=timeseries.shape[0],
        hidden_dim=256,
        latent_dim=64
    )
    model.eval()

    # Convert to torch
    with torch.no_grad():
        x = torch.FloatTensor(timeseries).unsqueeze(0)  # [1 × n_regions × time]
        features = model(x)  # [1 × latent_dim]

    return features.numpy()
```

**Option 2: Simpler Baseline - PCA on Functional Connectivity**

```python
from sklearn.decomposition import PCA
from nilearn.connectome import ConnectivityMeasure

def extract_functional_features_pca(timeseries: np.ndarray, n_components=50) -> np.ndarray:
    """Extract functional connectivity features via PCA.

    Args:
        timeseries: [n_regions × n_timepoints]
        n_components: Number of principal components

    Returns:
        features: [n_regions × n_components]
    """
    # Compute correlation matrix
    correlation_measure = ConnectivityMeasure(kind='correlation')
    connectivity = correlation_measure.fit_transform([timeseries.T])[0]  # [n_regions × n_regions]

    # PCA on connectivity patterns
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(connectivity)  # [n_regions × n_components]

    return features
```

### Step 3: Build Functional Graph from Features

```python
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def build_functional_graph(features: np.ndarray, threshold=0.3) -> nx.Graph:
    """Build graph from functional features.

    Args:
        features: [n_regions × n_features]
        threshold: Similarity threshold for edges

    Returns:
        G_functional: NetworkX graph with functional connectivity
    """
    # Compute pairwise similarity
    similarity = cosine_similarity(features)  # [n_regions × n_regions]

    # Build graph
    G_functional = nx.Graph()
    n_regions = features.shape[0]

    for i in range(n_regions):
        G_functional.add_node(i, features=features[i])

    # Add edges above threshold
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if similarity[i, j] > threshold:
                G_functional.add_edge(i, j, weight=similarity[i, j])

    return G_functional
```

### Step 4: Detect Functional Communities

```python
async def detect_functional_communities(
    G_functional: nx.Graph,
    target_systems: int = 10
) -> Dict[int, int]:
    """Detect functional systems using cluster-editing-vs.

    Args:
        G_functional: Functional connectivity graph
        target_systems: Expected number of functional systems

    Returns:
        communities: Node → community mapping
    """
    from backend.visualization.hierarchical_brain_model import detect_communities_auto

    communities = await detect_communities_auto(
        G_functional,
        target_clusters=target_systems,
        method="cluster_editing_vs",
        use_gpu=True  # Use GPU for large graphs
    )

    return communities
```

### Step 5: Create Semantic Hierarchy

```python
def build_semantic_hierarchy(
    G_functional: nx.Graph,
    communities: Dict[int, int],
    features: np.ndarray
) -> Dict[str, nx.Graph]:
    """Build semantic (functional) hierarchy.

    Returns:
        semantic_graphs: Multi-level functional hierarchy
    """
    from backend.visualization.hierarchical_brain_model import (
        contract_clusters_to_supernodes
    )

    # Level 3: Individual regions (features)
    G_features = G_functional.copy()

    # Level 2: Functional networks (contracted communities)
    G_networks, network_pos, network_members = contract_clusters_to_supernodes(
        G_features, communities, features
    )

    # Level 1: Functional systems (backbone)
    # Use same backbone detection as Phase 7
    from backend.visualization.hierarchical_brain_model import compute_backbone_hubs
    backbone = await compute_backbone_hubs(G_networks, r=2)

    semantic_graphs = {
        "features": G_features,       # Level 3
        "networks": G_networks,       # Level 2
        "systems": backbone           # Level 1
    }

    return semantic_graphs
```

## Complete Phase 8 Pipeline

```python
async def phase8_semantic_hierarchy(
    fmri_path: str,
    atlas_mask: str,
    target_systems: int = 10
):
    """Complete Phase 8: Extract semantic hierarchy from fMRI.

    Args:
        fmri_path: Path to 4D fMRI NIfTI file
        atlas_mask: D99 atlas mask (from Phase 7)
        target_systems: Expected functional systems

    Returns:
        semantic_graphs: Multi-level functional hierarchy
        features: Extracted diffusion CNN features
    """
    print("=" * 70)
    print("Phase 8: Semantic Hierarchy from 4D MRI")
    print("=" * 70)
    print()

    # Step 1: Preprocess fMRI
    print("Step 1: Preprocessing 4D fMRI data...")
    timeseries = preprocess_fmri(fmri_path, atlas_mask)
    print(f"  Extracted {timeseries.shape[0]} regions × {timeseries.shape[1]} timepoints")
    print()

    # Step 2: Extract features
    print("Step 2: Extracting diffusion CNN features...")
    features = extract_diffusion_features(timeseries)
    print(f"  Features: {features.shape}")
    print()

    # Step 3: Build functional graph
    print("Step 3: Building functional connectivity graph...")
    G_functional = build_functional_graph(features, threshold=0.3)
    print(f"  Graph: {G_functional.number_of_nodes()} nodes, {G_functional.number_of_edges()} edges")
    print()

    # Step 4: Detect communities
    print("Step 4: Detecting functional communities...")
    communities = await detect_functional_communities(G_functional, target_systems)
    print(f"  Found {len(set(communities.values()))} functional systems")
    print()

    # Step 5: Build hierarchy
    print("Step 5: Building semantic hierarchy...")
    semantic_graphs = build_semantic_hierarchy(G_functional, communities, features)
    print("  ✅ Semantic hierarchy complete")
    print()

    return semantic_graphs, features
```

## Integration with Phase 7 (Syntactic Hierarchy)

**Phase 9 Preview: Bipartite Mapping**

```python
def create_structure_function_mapping(
    G_anatomical: nx.Graph,  # From Phase 7
    G_functional: nx.Graph,  # From Phase 8
    region_ids: List[int]    # Shared node IDs
) -> nx.Graph:
    """Create bipartite graph mapping anatomy ↔ function.

    Args:
        G_anatomical: Syntactic (anatomical) graph
        G_functional: Semantic (functional) graph
        region_ids: Brain region IDs shared by both

    Returns:
        G_bipartite: Bipartite graph with structure-function edges
    """
    G_bipartite = nx.Graph()

    # Add anatomical nodes (left partition)
    for node in G_anatomical.nodes():
        G_bipartite.add_node(
            f"anat_{node}",
            type="anatomical",
            region=node,
            bipartite=0
        )

    # Add functional nodes (right partition)
    for node in G_functional.nodes():
        G_bipartite.add_node(
            f"func_{node}",
            type="functional",
            region=node,
            bipartite=1
        )

    # Add bipartite edges (anatomy ↔ function)
    for region_id in region_ids:
        if region_id in G_anatomical and region_id in G_functional:
            # Edge weight = correlation between anatomical and functional features
            G_bipartite.add_edge(
                f"anat_{region_id}",
                f"func_{region_id}",
                weight=1.0  # Perfect mapping for same region
            )

    return G_bipartite
```

## Expected Results

**Semantic Hierarchy from Real Data:**
- **Level 3 (Features)**: 100 regions with learned functional features
- **Level 2 (Networks)**: 10-15 functional systems (visual, motor, default mode, etc.)
- **Level 1 (Systems)**: 3-5 backbone hubs (primary integration zones)

**Key Metrics:**
- **Modularity**: > 0.3 (functional systems should be distinct)
- **Silhouette score**: > 0.5 (feature clustering quality)
- **Explained variance**: > 80% (PCA on connectivity)

## Next Steps

1. **Acquire macaque fMRI data** or use public datasets:
   - [PRIME-DE](http://fcon_1000.projects.nitrc.org/indi/indiPRIME.html) - Primate fMRI repository
   - [NHP-BIDS](https://github.com/PRIME-RE/dataset_descriptions) - Non-human primate BIDS

2. **Train/adapt diffusion CNN model**:
   - Fine-tune on macaque fMRI (transfer from human models)
   - Or use simpler baseline (PCA on functional connectivity)

3. **Validate functional systems**:
   - Compare with known macaque functional networks
   - Visual system: V1, V2, V4, MT
   - Motor system: M1, PMd, PMv, SMA

4. **Implement Phase 9**: Bipartite structure-function mapping

## References

Add to `references.bib`:

```bibtex
@article{benedict2023diffusion,
  title={Diffusion Models for Medical Imaging},
  author={Benedict, [First Name] and others},
  journal={[Journal]},
  year={2023},
  note={Diffusion CNN for fMRI feature extraction}
}

@article{milham2018prime,
  title={An open resource for non-human primate imaging},
  author={Milham, Michael P and Ai, Lei and Koo, Bonhwang and others},
  journal={Neuron},
  volume={100},
  number={1},
  pages={61--74},
  year={2018},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2018.08.039}
}
```

---

**Status**: Phase 8 design complete, awaiting real fMRI data and diffusion CNN model selection.
