"""D99 v2.0 Macaque Brain Atlas implementation.

References:
    Saleem KS, Logothetis NK (2012). A Combined MRI and Histology Atlas 
    of the Rhesus Monkey Brain in Stereotaxic Coordinates. Academic Press.
    
    Reveley C et al. (2017). Three-Dimensional Digital Template Atlas 
    of the Macaque Brain. Cereb Cortex.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base_atlas import BrainAtlas, BrainRegion

# Optional imports - graceful degradation
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    nib = None


class D99Atlas(BrainAtlas):
    """D99 v2.0 Macaque Atlas implementation.
    
    The D99 atlas provides 368 brain regions:
    - 149 cortical regions
    - 219 subcortical regions
    
    Data location: data/atlases/D99_v2.0_dist/
    """
    
    ATLAS_DIR = Path(__file__).parent.parent.parent / "data" / "atlases" / "D99_v2.0_dist"
    
    def __init__(self):
        """Initialize D99 atlas."""
        self._atlas_img = None
        self._atlas_data = None
        self._template_img = None
        self._labels: List[BrainRegion] = []
        self._id_to_region = {}
        self._name_to_id = {}
        self._abbrev_to_id = {}
        self._loaded = False
        
    def _ensure_loaded(self):
        """Lazy load atlas data."""
        if self._loaded:
            return
            
        if not HAS_NIBABEL:
            raise ImportError("nibabel required for D99 atlas. Install with: pip install nibabel")
            
        if not self.ATLAS_DIR.exists():
            raise FileNotFoundError(
                f"D99 atlas not found at {self.ATLAS_DIR}. "
                "Install with: cd data/atlases && curl -O https://afni.nimh.nih.gov/pub/dist/atlases/macaque/D99_Saleem/D99_v2.0_dist.tgz && tar -xvf D99_v2.0_dist.tgz"
            )
        
        # Load atlas volume
        atlas_path = self.ATLAS_DIR / "D99_atlas_v2.0.nii.gz"
        if atlas_path.exists():
            self._atlas_img = nib.load(str(atlas_path))
            self._atlas_data = self._atlas_img.get_fdata().astype(np.int32)
        
        # Load template
        template_path = self.ATLAS_DIR / "D99_template.nii.gz"
        if template_path.exists():
            self._template_img = nib.load(str(template_path))
        
        # Load labels
        labels_path = self.ATLAS_DIR / "D99_v2.0_labels_semicolon.txt"
        if labels_path.exists():
            self._labels = self._parse_labels(labels_path)
            
            # Build lookup tables
            for region in self._labels:
                self._id_to_region[region.id] = region
                self._name_to_id[region.name.lower()] = region.id
                self._abbrev_to_id[region.abbreviation.lower()] = region.id
        
        self._loaded = True
    
    def _parse_labels(self, labels_path: Path) -> List[BrainRegion]:
        """Parse D99 labels file (semicolon-separated)."""
        regions = []
        
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ';' not in line:
                    continue
                    
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                    
                try:
                    region_id = int(parts[0].strip())
                    abbrev = parts[1].strip()
                    name = parts[2].strip() if len(parts) > 2 else abbrev
                    
                    # Determine type from ID ranges (approximate based on D99 structure)
                    # IDs 1-149 are generally cortical, 150+ subcortical
                    if region_id <= 149:
                        region_type = "cortical"
                    else:
                        region_type = "subcortical"
                    
                    # Infer functional system from name/abbreviation
                    functional_systems = self._infer_functional_systems(name, abbrev)
                    
                    regions.append(BrainRegion(
                        id=region_id,
                        name=name,
                        abbreviation=abbrev,
                        hemisphere="bilateral",
                        region_type=region_type,
                        functional_systems=functional_systems,
                    ))
                except (ValueError, IndexError):
                    continue
                    
        return regions
    
    def _infer_functional_systems(self, name: str, abbrev: str) -> List[str]:
        """Infer functional systems from region name/abbreviation."""
        systems = []
        combined = f"{name} {abbrev}".lower()
        
        if any(x in combined for x in ['v1', 'v2', 'v3', 'v4', 'mt', 'visual', 'occipital', 'calcarine']):
            systems.append("visual")
        if any(x in combined for x in ['a1', 'audit', 'temporal']):
            systems.append("auditory")
        if any(x in combined for x in ['motor', 'm1', 'precentral', 'sma', 'premotor']):
            systems.append("motor")
        if any(x in combined for x in ['somato', 's1', 'postcentral']):
            systems.append("somatosensory")
        if any(x in combined for x in ['hippoc', 'amygd', 'cingul', 'limbic', 'entorhinal']):
            systems.append("limbic")
        if any(x in combined for x in ['prefrontal', 'pfc', 'dlpfc', 'orbito']):
            systems.append("prefrontal")
            
        return systems
    
    @property
    def species(self) -> str:
        return "macaque"
    
    @property
    def name(self) -> str:
        return "D99"
    
    @property
    def version(self) -> str:
        return "2.0"
    
    @property
    def citation(self) -> str:
        return (
            "Saleem KS, Logothetis NK (2012). A Combined MRI and Histology Atlas "
            "of the Rhesus Monkey Brain in Stereotaxic Coordinates. Academic Press. "
            "Reveley C et al. (2017). Cereb Cortex."
        )
    
    def get_region(
        self, 
        query: str, 
        query_type: str = "name"
    ) -> Optional[BrainRegion]:
        """Look up region by name, abbreviation, or ID."""
        self._ensure_loaded()
        
        query_lower = query.lower().strip()
        region_id = None
        
        if query_type == "name":
            region_id = self._name_to_id.get(query_lower)
        elif query_type == "abbreviation":
            region_id = self._abbrev_to_id.get(query_lower)
        elif query_type == "id":
            try:
                region_id = int(query)
            except ValueError:
                pass
        else:  # auto
            # Try all lookup methods
            region_id = (
                self._name_to_id.get(query_lower) or 
                self._abbrev_to_id.get(query_lower)
            )
            if region_id is None:
                try:
                    region_id = int(query)
                except ValueError:
                    pass
        
        if region_id is not None and region_id in self._id_to_region:
            region = self._id_to_region[region_id]
            
            # Compute centroid and volume if not cached
            if region.centroid is None and self._atlas_data is not None:
                region.centroid = self._compute_centroid(region_id)
                region.volume_mm3 = self._compute_volume(region_id)
            
            return region
            
        return None
    
    def get_region_at_coordinate(
        self, 
        x: float, 
        y: float, 
        z: float,
        space: str = "native"
    ) -> Optional[BrainRegion]:
        """Get region at specified coordinate."""
        self._ensure_loaded()
        
        if self._atlas_data is None or self._atlas_img is None:
            return None
        
        # Convert world to voxel coordinates
        affine_inv = np.linalg.inv(self._atlas_img.affine)
        voxel = affine_inv @ np.array([x, y, z, 1])
        i, j, k = int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
        
        # Bounds check
        shape = self._atlas_data.shape
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            return None
        
        region_id = self._atlas_data[i, j, k]
        if region_id > 0:
            return self._id_to_region.get(int(region_id))
            
        return None
    
    def _compute_centroid(self, region_id: int) -> Tuple[float, float, float]:
        """Compute centroid of region in world coordinates."""
        if self._atlas_data is None:
            return (0.0, 0.0, 0.0)
            
        mask = self._atlas_data == region_id
        indices = np.argwhere(mask)
        
        if len(indices) == 0:
            return (0.0, 0.0, 0.0)
        
        centroid_voxel = indices.mean(axis=0)
        centroid_world = self._atlas_img.affine @ np.append(centroid_voxel, 1)
        return tuple(float(x) for x in centroid_world[:3])
    
    def _compute_volume(self, region_id: int) -> float:
        """Compute volume in mm³."""
        if self._atlas_data is None:
            return 0.0
            
        mask = self._atlas_data == region_id
        voxel_count = np.sum(mask)
        voxel_vol = abs(np.linalg.det(self._atlas_img.affine[:3, :3]))
        return float(voxel_count * voxel_vol)
    
    def list_regions(
        self, 
        region_type: Optional[str] = None,
        hierarchy_level: Optional[int] = None,
        search_pattern: Optional[str] = None
    ) -> List[BrainRegion]:
        """List all regions with optional filtering."""
        self._ensure_loaded()
        
        regions = list(self._labels)
        
        if region_type:
            regions = [r for r in regions if r.region_type == region_type]
        
        if search_pattern:
            pattern = re.compile(search_pattern, re.IGNORECASE)
            regions = [r for r in regions if pattern.search(r.name) or pattern.search(r.abbreviation)]
        
        return regions
    
    def generate_mask(
        self, 
        region_ids: List[int],
        output_path: str,
        reference_volume: Optional[str] = None,
        dilate_mm: float = 0,
        hemisphere: str = "both"
    ) -> str:
        """Generate NIfTI mask for specified regions."""
        self._ensure_loaded()
        
        if self._atlas_data is None:
            raise RuntimeError("Atlas data not loaded")
        
        # Create binary mask
        mask = np.isin(self._atlas_data, region_ids).astype(np.uint8)
        
        # Dilate if requested
        if dilate_mm > 0:
            from scipy.ndimage import binary_dilation
            # Convert mm to voxels (approximate)
            voxel_size = abs(self._atlas_img.affine[0, 0])
            iterations = int(dilate_mm / voxel_size)
            if iterations > 0:
                mask = binary_dilation(mask, iterations=iterations).astype(np.uint8)
        
        # Handle hemisphere
        if hemisphere != "both" and self._atlas_data is not None:
            center_x = self._atlas_data.shape[0] // 2
            if hemisphere == "left":
                mask[center_x:, :, :] = 0
            elif hemisphere == "right":
                mask[:center_x, :, :] = 0
        
        # Resample to reference if provided
        if reference_volume and HAS_NIBABEL:
            # TODO: Implement resampling
            pass
        
        # Save mask
        mask_img = nib.Nifti1Image(mask, self._atlas_img.affine)
        nib.save(mask_img, output_path)
        
        return output_path
    
    def get_neighbors(self, region_id: int) -> List[BrainRegion]:
        """Get neighboring regions that share boundary voxels."""
        self._ensure_loaded()
        
        if self._atlas_data is None:
            return []
        
        from scipy.ndimage import binary_dilation
        
        mask = self._atlas_data == region_id
        dilated = binary_dilation(mask)
        boundary = dilated & ~mask
        
        neighbor_ids = np.unique(self._atlas_data[boundary])
        neighbor_ids = neighbor_ids[neighbor_ids > 0]
        neighbor_ids = neighbor_ids[neighbor_ids != region_id]
        
        return [
            self._id_to_region[int(nid)] 
            for nid in neighbor_ids 
            if int(nid) in self._id_to_region
        ]
