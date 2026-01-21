"""QEC Tensor Service - Access merge2docs tensor via HTTP/MCP.

This service provides access to the merge2docs 3D QEC tensor (Functor × Domain × Level)
and helps bootstrap the brain-specific tensor.

Architecture:
- merge2docs: 5 functors × 24 domains × 4 levels = 480 cells (20-30 populated)
- twosphere: 6 functors × 100 regions × 3 scales = 1,800 cells (adaptive loading)

Bootstrap strategy:
1. Download merge2docs tensor corpus (56MB) once
2. Extract learned patterns (F_i hierarchy, r-IDS, cross-training)
3. Adapt to brain: Map functors → brain modalities, domains → regions
4. Build brain-specific tensor with recursive cross-training
5. Expose back to merge2docs for integration

F_i Functor Mapping:
    merge2docs → brain
    wisdom     → behavior    (high-level understanding)
    papers     → function    (what it computes)
    code       → anatomy     (implementation/structure)
    testing    → electro     (validation/dynamics)
    git        → genetics    (version control/heritage)
    [new]      → pathology   (disease/errors)
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import json
import requests
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Functor Hierarchy Functions
# =============================================================================

def map_functor(merge2docs_functor: str) -> str:
    """Map merge2docs functor to brain functor.

    Args:
        merge2docs_functor: Source functor (wisdom, papers, code, testing, git)

    Returns:
        Brain functor (anatomy, function, electro, genetics, behavior, pathology)
    """
    mapping = {
        "wisdom": "behavior",    # F0: High-level understanding → Task relevance
        "papers": "function",    # F1: Research → What region computes
        "code": "anatomy",       # F2: Implementation → Structure
        "testing": "electro",    # F3: Validation → Dynamics
        "git": "genetics"        # F5: Version control → Heritage
        # pathology is brain-specific (no merge2docs equivalent)
    }
    return mapping.get(merge2docs_functor, merge2docs_functor)


def can_teach(source_functor: str, target_functor: str) -> bool:
    """Check if source functor can teach target functor in F_i hierarchy.

    Brain F_i Hierarchy (brain-research-v1):
    - F0: anatomy (structure)
    - F1: function (computation)
    - F2: electro (dynamics)
    - F3: genetics (heritage)
    - F4: behavior (task relevance)
    - F5: pathology (disease markers)

    Teaching Rule: Higher abstraction (lower index) teaches lower

    Args:
        source_functor: Source functor name
        target_functor: Target functor name

    Returns:
        True if source can teach target, False otherwise
    """
    hierarchy = [
        "anatomy",      # F0: Structure
        "function",     # F1: Computation
        "electro",      # F2: Dynamics
        "genetics",     # F3: Heritage
        "behavior",     # F4: Task relevance
        "pathology"     # F5: Disease markers
    ]

    if source_functor not in hierarchy or target_functor not in hierarchy:
        return False

    source_idx = hierarchy.index(source_functor)
    target_idx = hierarchy.index(target_functor)

    # Higher abstraction (lower index) teaches lower
    # Reflexive: F can teach itself (identity morphism)
    return source_idx <= target_idx


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QECTensorConfig:
    """Configuration for QEC tensor access."""

    # merge2docs service
    merge2docs_url: str = "http://localhost:8091"

    # Bootstrap endpoints
    download_endpoint: str = "/qec/tensor/corpus/download"
    cells_list_endpoint: str = "/qec/tensor/cells"
    brain_mapping_endpoint: str = "/qec/brain_regions/mapping"

    # Local storage
    cache_dir: Path = Path(__file__).parent.parent.parent.parent / "cache" / "qec_tensor"
    corpus_filename: str = "merge2docs_tensor_corpus.pkl"

    # Tensor dimensions (merge2docs)
    merge2docs_functors: List[str] = None
    merge2docs_domains: List[str] = None
    merge2docs_levels: List[str] = None

    # Brain tensor dimensions (twosphere)
    brain_functors: List[str] = None
    brain_regions: int = 100  # D99 cortical regions (will expand to 368)
    brain_scales: List[str] = None

    # Functor mapping
    functor_mapping: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.merge2docs_functors is None:
            self.merge2docs_functors = ["wisdom", "papers", "code", "testing", "git"]

        if self.merge2docs_domains is None:
            self.merge2docs_domains = [
                "llm", "machine_learning", "combinatorial", "mathematics", "statistics",
                "physics", "molecular_bio", "bioinformatics", "economics", "theology",
                "classic", "memetic", "meta_analysis", "natec", "qpp", "policies",
                "software_eng", "theoretical", "papers", "manuals", "designs",
                "wisdom", "archive", "general"
            ]

        if self.merge2docs_levels is None:
            self.merge2docs_levels = ["para", "section", "chapter", "document"]

        if self.brain_functors is None:
            self.brain_functors = ["anatomy", "function", "electro", "genetics", "behavior", "pathology"]

        if self.brain_scales is None:
            self.brain_scales = ["column", "region", "system"]

        if self.functor_mapping is None:
            # Map merge2docs F_i → brain F_i
            self.functor_mapping = {
                "wisdom": "behavior",    # F0: High-level understanding → Task relevance
                "papers": "function",    # F1: Research → What brain computes
                "code": "anatomy",       # F2: Implementation → Structure
                "testing": "electro",    # F3: Validation → Neural dynamics
                "git": "genetics",       # F5: Version control → Genetic heritage
                # brain-specific
                "pathology": "pathology" # New: Disease markers
            }

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def corpus_path(self) -> Path:
        """Get full path to corpus file."""
        return self.cache_dir / self.corpus_filename


# =============================================================================
# QEC Tensor Client
# =============================================================================

class QECTensorClient:
    """Client for accessing merge2docs QEC tensor via HTTP."""

    def __init__(self, config: Optional[QECTensorConfig] = None):
        """Initialize QEC tensor client.

        Args:
            config: Configuration (default: QECTensorConfig())
        """
        self.config = config or QECTensorConfig()
        self.session = requests.Session()
        self._corpus_cached = False

    async def bootstrap_from_merge2docs(self, force_download: bool = False) -> Dict:
        """Bootstrap brain tensor by downloading merge2docs corpus.

        This is a ONE-TIME operation that downloads the full merge2docs tensor (56MB).

        Args:
            force_download: If True, re-download even if cached

        Returns:
            corpus: Full tensor corpus with metadata
        """
        logger.info("=" * 70)
        logger.info("Bootstrapping Brain QEC Tensor from merge2docs")
        logger.info("=" * 70)

        # Check cache first
        if not force_download and self.config.corpus_path.exists():
            logger.info(f"✅ Corpus already cached: {self.config.corpus_path}")
            logger.info("   Use force_download=True to re-download")
            return self._load_corpus_from_cache()

        # Download corpus
        logger.info("Step 1: Downloading merge2docs tensor corpus (56MB)...")
        url = f"{self.config.merge2docs_url}{self.config.download_endpoint}"

        try:
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Save to cache
            with open(self.config.corpus_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"   ✅ Downloaded: {self.config.corpus_path}")
            logger.info(f"   Size: {self.config.corpus_path.stat().st_size / 1024 / 1024:.1f} MB")

        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Download failed: {e}")
            logger.error(f"   Ensure merge2docs service is running at {self.config.merge2docs_url}")
            raise

        # Load and return
        return self._load_corpus_from_cache()

    def _load_corpus_from_cache(self) -> Dict:
        """Load corpus from local cache."""
        import pickle

        logger.info("Step 2: Loading corpus from cache...")

        with open(self.config.corpus_path, 'rb') as f:
            corpus = pickle.load(f)

        logger.info(f"   ✅ Loaded {len(corpus.get('cells', {}))} tensor cells")
        logger.info(f"   Functors: {corpus.get('functors', [])}")
        logger.info(f"   Domains: {len(corpus.get('domains', []))} domains")
        logger.info(f"   Levels: {corpus.get('levels', [])}")

        self._corpus_cached = True
        return corpus

    async def list_available_cells(self) -> List[Dict]:
        """List all populated tensor cells.

        Returns:
            cells: List of cell metadata
        """
        url = f"{self.config.merge2docs_url}{self.config.cells_list_endpoint}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            cells = data.get("cells", [])
            logger.info(f"Retrieved {len(cells)} tensor cells from merge2docs")

            return cells

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list cells: {e}")
            return []

    async def get_brain_region_mapping(self) -> Dict[str, str]:
        """Get suggested mapping from merge2docs domains → brain regions.

        This helps bootstrap the initial brain tensor by finding analogies:
        - "molecular_bio" domain → hippocampus (memory)
        - "combinatorial" domain → PFC (executive function)
        - etc.

        Returns:
            mapping: {merge2docs_domain → brain_region}
        """
        url = f"{self.config.merge2docs_url}{self.config.brain_mapping_endpoint}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            mapping = data.get("mapping", {})
            logger.info(f"Retrieved domain→region mapping for {len(mapping)} domains")

            return mapping

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get brain mapping: {e}")
            # Return empty mapping if endpoint not available yet
            return {}

    async def extract_learned_patterns(self, corpus: Dict) -> Dict:
        """Extract learned patterns from merge2docs corpus.

        This extracts:
        - F_i hierarchy teaching rules
        - r-IDS computation parameters
        - Cross-training strategies
        - Syndrome detection thresholds

        Args:
            corpus: Loaded corpus from bootstrap

        Returns:
            patterns: Learned patterns to apply to brain tensor
        """
        logger.info("Step 3: Extracting learned patterns...")

        patterns = {
            "fi_hierarchy": {},
            "rids_params": {},
            "cross_training": {},
            "syndromes": {}
        }

        # Extract F_i hierarchy rules
        if "fi_config" in corpus:
            fi_config = corpus["fi_config"]
            patterns["fi_hierarchy"] = {
                "levels": fi_config.get("levels", ["F0", "F1", "F2", "F3", "F4", "F5"]),
                "teaching_rules": fi_config.get("teaching_rules", {}),
                "direction_aware": fi_config.get("direction_aware", True)
            }
            logger.info(f"   ✅ F_i hierarchy: {len(patterns['fi_hierarchy']['levels'])} levels")

        # Extract r-IDS parameters
        if "rids_config" in corpus:
            patterns["rids_params"] = {
                "r": corpus["rids_config"].get("r", 4),
                "method": corpus["rids_config"].get("method", "greedy"),
                "coverage_threshold": corpus["rids_config"].get("coverage_threshold", 0.95)
            }
            logger.info(f"   ✅ r-IDS: r={patterns['rids_params']['r']}")

        # Extract cross-training strategy
        if "cross_training_config" in corpus:
            patterns["cross_training"] = {
                "rounds": corpus["cross_training_config"].get("rounds", 5),
                "learning_rate": corpus["cross_training_config"].get("lr", 1e-4),
                "bridge_threshold": corpus["cross_training_config"].get("bridge_threshold", 0.3)
            }
            logger.info(f"   ✅ Cross-training: {patterns['cross_training']['rounds']} rounds")

        # Extract syndrome thresholds
        if "syndrome_config" in corpus:
            patterns["syndromes"] = {
                "threshold": corpus["syndrome_config"].get("threshold", 0.1),
                "types": corpus["syndrome_config"].get("types", ["cross_functor", "cross_domain"])
            }
            logger.info(f"   ✅ Syndromes: threshold={patterns['syndromes']['threshold']}")

        return patterns

    async def adapt_to_brain(self, corpus: Dict, patterns: Dict) -> Dict:
        """Adapt merge2docs tensor to brain-specific tensor.

        This creates the initial brain tensor structure by:
        1. Mapping functors (wisdom→behavior, papers→function, etc.)
        2. Replacing domains with brain regions
        3. Adjusting levels to brain scales (column, region, system)
        4. Preserving learned patterns (r-IDS, cross-training, F_i)

        Args:
            corpus: merge2docs tensor corpus
            patterns: Extracted learned patterns

        Returns:
            brain_tensor: Initial brain tensor structure
        """
        logger.info("Step 4: Adapting to brain-specific tensor...")

        brain_tensor = {
            "metadata": {
                "source": "merge2docs_bootstrap",
                "bootstrap_date": str(Path(self.config.corpus_path).stat().st_mtime),
                "merge2docs_version": corpus.get("version", "unknown")
            },
            "config": {
                "functors": self.config.brain_functors,
                "regions": self.config.brain_regions,
                "scales": self.config.brain_scales,
                "r": patterns["rids_params"].get("r", 4)
            },
            "functor_mapping": self.config.functor_mapping,
            "learned_patterns": patterns,
            "cells": {}
        }

        logger.info(f"   Brain tensor structure:")
        logger.info(f"     Functors: {len(brain_tensor['config']['functors'])}")
        logger.info(f"     Regions: {brain_tensor['config']['regions']}")
        logger.info(f"     Scales: {len(brain_tensor['config']['scales'])}")
        logger.info(f"     Total cells: {len(brain_tensor['config']['functors']) * brain_tensor['config']['regions'] * len(brain_tensor['config']['scales'])}")

        # Map merge2docs cells → brain cells (as templates)
        merge2docs_cells = corpus.get("cells", {})

        for cell_name, cell_data in merge2docs_cells.items():
            # Parse cell name: {functor}_{domain}_{level}
            parts = cell_name.split("_")
            if len(parts) != 3:
                continue

            merge2docs_functor, domain, level = parts

            # Map functor
            brain_functor = self.config.functor_mapping.get(merge2docs_functor)
            if not brain_functor:
                continue

            # Store as template for this functor type
            if brain_functor not in brain_tensor["cells"]:
                brain_tensor["cells"][brain_functor] = []

            brain_tensor["cells"][brain_functor].append({
                "template_source": cell_name,
                "learned_features": cell_data.get("features"),
                "rids_connections": cell_data.get("rids_connections"),
                "training_history": cell_data.get("training_history")
            })

        logger.info(f"   ✅ Mapped {sum(len(v) for v in brain_tensor['cells'].values())} cell templates")

        return brain_tensor

    async def save_brain_tensor(self, brain_tensor: Dict, filename: str = "brain_tensor_bootstrapped.pkl"):
        """Save brain tensor to local cache.

        Args:
            brain_tensor: Brain tensor structure
            filename: Output filename
        """
        import pickle

        output_path = self.config.cache_dir / filename

        with open(output_path, 'wb') as f:
            pickle.dump(brain_tensor, f)

        logger.info(f"✅ Saved brain tensor: {output_path}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        return output_path


# =============================================================================
# High-Level API
# =============================================================================

async def bootstrap_brain_tensor(
    force_download: bool = False,
    config: Optional[QECTensorConfig] = None
) -> Dict:
    """Bootstrap brain tensor from merge2docs (high-level API).

    This is the main entry point for one-time bootstrap.

    Args:
        force_download: Re-download even if cached
        config: Optional configuration

    Returns:
        brain_tensor: Bootstrapped brain tensor ready for training
    """
    client = QECTensorClient(config)

    # Step 1-2: Download/load corpus
    corpus = await client.bootstrap_from_merge2docs(force_download=force_download)

    # Step 3: Extract learned patterns
    patterns = await client.extract_learned_patterns(corpus)

    # Step 4: Adapt to brain
    brain_tensor = await client.adapt_to_brain(corpus, patterns)

    # Step 5: Save
    await client.save_brain_tensor(brain_tensor)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Bootstrap Complete!")
    logger.info("=" * 70)
    logger.info(f"Brain tensor ready with {len(brain_tensor['config']['functors'])} functors")
    logger.info(f"Next steps:")
    logger.info(f"  1. Load D99 atlas regions")
    logger.info(f"  2. Populate {brain_tensor['config']['regions']} region tensors")
    logger.info(f"  3. Train with recursive cross-training")
    logger.info(f"  4. Expose back to merge2docs via MCP")
    logger.info("")

    return brain_tensor


async def list_merge2docs_cells(config: Optional[QECTensorConfig] = None) -> List[Dict]:
    """List available merge2docs tensor cells.

    Args:
        config: Optional configuration

    Returns:
        cells: List of cell metadata
    """
    client = QECTensorClient(config)
    return await client.list_available_cells()


# =============================================================================
# Example Usage
# =============================================================================

async def demo_bootstrap():
    """Demo: Bootstrap brain tensor from merge2docs."""

    print("=" * 70)
    print("Demo: Bootstrap Brain Tensor from merge2docs")
    print("=" * 70)
    print()

    # Bootstrap
    brain_tensor = await bootstrap_brain_tensor(force_download=False)

    # Inspect
    print()
    print("Bootstrapped brain tensor:")
    print(f"  Functors: {brain_tensor['config']['functors']}")
    print(f"  Regions: {brain_tensor['config']['regions']}")
    print(f"  Scales: {brain_tensor['config']['scales']}")
    print(f"  r-IDS: r={brain_tensor['config']['r']}")
    print()

    # Check functor mapping
    print("Functor mapping (merge2docs → brain):")
    for merge2docs_f, brain_f in brain_tensor['functor_mapping'].items():
        print(f"  {merge2docs_f:10s} → {brain_f}")
    print()

    # Check learned patterns
    patterns = brain_tensor['learned_patterns']
    print("Learned patterns from merge2docs:")
    print(f"  F_i levels: {patterns['fi_hierarchy'].get('levels', [])}")
    print(f"  r-IDS r: {patterns['rids_params'].get('r')}")
    print(f"  Cross-training rounds: {patterns['cross_training'].get('rounds')}")
    print(f"  Syndrome threshold: {patterns['syndromes'].get('threshold')}")
    print()


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_bootstrap())
