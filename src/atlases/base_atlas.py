"""Abstract base class for brain atlases."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class BrainRegion:
    """Represents a brain region in an atlas."""
    
    id: int
    name: str
    abbreviation: str
    hemisphere: str = "bilateral"  # 'left', 'right', 'bilateral'
    region_type: str = "unknown"   # 'cortical', 'subcortical', 'cerebellar', 'white_matter'
    parent_id: Optional[int] = None
    centroid: Optional[Tuple[float, float, float]] = None
    volume_mm3: Optional[float] = None
    color_rgb: Optional[Tuple[int, int, int]] = None
    functional_systems: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "abbreviation": self.abbreviation,
            "hemisphere": self.hemisphere,
            "region_type": self.region_type,
            "parent_id": self.parent_id,
            "centroid": list(self.centroid) if self.centroid else None,
            "volume_mm3": self.volume_mm3,
            "color_rgb": list(self.color_rgb) if self.color_rgb else None,
            "functional_systems": self.functional_systems,
        }


class BrainAtlas(ABC):
    """Abstract base class for brain atlases.
    
    Implementations must provide methods to:
    - Look up regions by name, abbreviation, ID, or coordinate
    - List all regions with optional filtering
    - Generate NIfTI masks for regions
    - Find neighboring regions
    """
    
    @property
    @abstractmethod
    def species(self) -> str:
        """Return species name (e.g., 'macaque', 'mouse', 'rat')."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return atlas name (e.g., 'D99', 'Allen_CCF', 'Waxholm')."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return atlas version."""
        pass
    
    @property
    def citation(self) -> str:
        """Return citation for the atlas."""
        return ""
    
    @abstractmethod
    def get_region(
        self, 
        query: str, 
        query_type: str = "name"
    ) -> Optional[BrainRegion]:
        """Look up a region by name, abbreviation, or ID.
        
        Args:
            query: Search string
            query_type: 'name', 'abbreviation', 'id', or 'auto'
            
        Returns:
            BrainRegion if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_region_at_coordinate(
        self, 
        x: float, 
        y: float, 
        z: float,
        space: str = "native"
    ) -> Optional[BrainRegion]:
        """Get region at specified coordinate.
        
        Args:
            x, y, z: Coordinates
            space: Coordinate space ('native', 'MNI', 'stereotaxic')
            
        Returns:
            BrainRegion if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_regions(
        self, 
        region_type: Optional[str] = None,
        hierarchy_level: Optional[int] = None,
        search_pattern: Optional[str] = None
    ) -> List[BrainRegion]:
        """List all regions with optional filtering.
        
        Args:
            region_type: Filter by type ('cortical', 'subcortical', etc.)
            hierarchy_level: Filter by hierarchy level (0=whole brain)
            search_pattern: Regex pattern to filter names
            
        Returns:
            List of BrainRegion objects
        """
        pass
    
    @abstractmethod
    def generate_mask(
        self, 
        region_ids: List[int],
        output_path: str,
        reference_volume: Optional[str] = None,
        dilate_mm: float = 0,
        hemisphere: str = "both"
    ) -> str:
        """Generate NIfTI mask for specified regions.
        
        Args:
            region_ids: List of region IDs to include
            output_path: Path to save output NIfTI
            reference_volume: Optional volume to match grid/resolution
            dilate_mm: Dilation in mm
            hemisphere: 'left', 'right', or 'both'
            
        Returns:
            Path to saved mask
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, region_id: int) -> List[BrainRegion]:
        """Get neighboring regions that share boundary voxels.
        
        Args:
            region_id: ID of region to find neighbors for
            
        Returns:
            List of neighboring BrainRegion objects
        """
        pass
    
    def get_hierarchy(self, region_id: int) -> Dict:
        """Get hierarchical context for a region.
        
        Args:
            region_id: ID of region
            
        Returns:
            Dict with 'parent' and 'children' keys
        """
        region = self.get_region(str(region_id), query_type="id")
        if not region:
            return {"parent": None, "children": []}
        
        # Get parent
        parent = None
        if region.parent_id:
            parent = self.get_region(str(region.parent_id), query_type="id")
        
        # Get children
        children = [r for r in self.list_regions() if r.parent_id == region_id]
        
        return {
            "parent": parent.to_dict() if parent else None,
            "children": [c.to_dict() for c in children]
        }
    
    def info(self) -> Dict:
        """Return atlas metadata."""
        all_regions = self.list_regions()
        cortical = [r for r in all_regions if r.region_type == "cortical"]
        subcortical = [r for r in all_regions if r.region_type == "subcortical"]
        
        return {
            "name": self.name,
            "version": self.version,
            "species": self.species,
            "citation": self.citation,
            "total_regions": len(all_regions),
            "cortical_regions": len(cortical),
            "subcortical_regions": len(subcortical),
        }
