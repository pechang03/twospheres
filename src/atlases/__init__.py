"""Brain atlas interfaces for multi-species neuroimaging.

Supports:
- Macaque: D99 v2.0, CHARM, SARM
- Mouse: Allen CCF v3
- Rat: Waxholm Space
- Rodent Eyes: Mouse Retina Cell Atlas (future)
"""

from .base_atlas import BrainAtlas, BrainRegion
from .d99_atlas import D99Atlas

__all__ = [
    'BrainAtlas',
    'BrainRegion', 
    'D99Atlas',
    'get_atlas',
]


def get_atlas(species: str, atlas_name: str = None) -> BrainAtlas:
    """Factory function to get atlas instance for species.
    
    Args:
        species: 'macaque', 'mouse', or 'rat'
        atlas_name: Optional specific atlas (e.g., 'D99', 'Allen_CCF', 'Waxholm')
        
    Returns:
        BrainAtlas instance
        
    Raises:
        ValueError: Unknown species/atlas
        NotImplementedError: Atlas not yet implemented
    """
    species = species.lower()
    
    if species == "macaque":
        if atlas_name is None or atlas_name.upper() == "D99":
            return D99Atlas()
        else:
            raise NotImplementedError(f"Macaque atlas '{atlas_name}' not yet implemented")
            
    elif species == "mouse":
        # from .allen_ccf_atlas import AllenCCFAtlas
        # return AllenCCFAtlas()
        raise NotImplementedError("Allen CCF atlas not yet implemented - download in progress")
        
    elif species == "rat":
        # from .waxholm_atlas import WaxholmAtlas
        # return WaxholmAtlas()
        raise NotImplementedError("Waxholm rat atlas not yet implemented - download in progress")
        
    else:
        raise ValueError(f"Unknown species: {species}. Supported: macaque, mouse, rat")
