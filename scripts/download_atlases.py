#!/usr/bin/env python3
"""Download brain atlases for the brain-atlas-mcp server.

Usage:
    python scripts/download_atlases.py --all
    python scripts/download_atlases.py --d99        # Macaque D99 v2.0
    python scripts/download_atlases.py --allen      # Mouse Allen CCF v3 (requires allensdk)
    python scripts/download_atlases.py --waxholm    # Rat Waxholm Space
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ATLAS_DIR = Path(__file__).parent.parent / "data" / "atlases"


def download_d99():
    """Download D99 macaque atlas from AFNI."""
    d99_dir = ATLAS_DIR / "D99_v2.0_dist"
    if d99_dir.exists() and (d99_dir / "D99_atlas_v2.0.nii.gz").exists():
        print("✅ D99 v2.0 already installed")
        return True
    
    print("📥 Downloading D99 v2.0 macaque atlas from AFNI...")
    d99_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://afni.nimh.nih.gov/pub/dist/atlases/macaque/D99_Saleem/D99_v2.0_dist.tgz"
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(d99_dir.parent / "D99_v2.0_dist.tgz"), url],
            check=True
        )
        subprocess.run(
            ["tar", "-xzf", "D99_v2.0_dist.tgz"],
            cwd=str(d99_dir.parent),
            check=True
        )
        os.remove(d99_dir.parent / "D99_v2.0_dist.tgz")
        print("✅ D99 v2.0 installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ D99 download failed: {e}")
        return False


def download_allen():
    """Download Allen CCF v3 mouse atlas using allensdk."""
    allen_dir = ATLAS_DIR / "Allen_CCF_mouse"
    allen_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        print("❌ allensdk not installed. Run: pip install allensdk")
        print("   Then re-run this script with --allen")
        return False
    
    print("📥 Downloading Allen CCF v3 mouse atlas via allensdk...")
    print("   This may take several minutes (downloads ~4GB)...")
    
    try:
        mcc = MouseConnectivityCache(
            manifest_file=str(allen_dir / 'mouse_connectivity_manifest.json'),
            resolution=25  # 25µm resolution (smaller download)
        )
        
        # Download the annotation volume (atlas labels)
        annotation, _ = mcc.get_annotation_volume()
        print(f"   Downloaded annotation volume: {annotation.shape}")
        
        # Download the template volume
        template, _ = mcc.get_template_volume()
        print(f"   Downloaded template volume: {template.shape}")
        
        # Get structure tree
        structure_tree = mcc.get_structure_tree()
        structures = structure_tree.get_structures_by_set_id([1])  # All structures
        print(f"   Downloaded {len(structures)} structure definitions")
        
        print("✅ Allen CCF v3 installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Allen CCF download failed: {e}")
        return False


def download_waxholm():
    """Download Waxholm Space rat atlas.
    
    Note: Waxholm requires manual download from NITRC due to licensing.
    """
    waxholm_dir = ATLAS_DIR / "Waxholm_rat"
    waxholm_dir.mkdir(parents=True, exist_ok=True)
    
    if list(waxholm_dir.glob("*.nii*")):
        print("✅ Waxholm atlas files found")
        return True
    
    print("📋 Waxholm Space Rat Atlas requires manual download:")
    print()
    print("   1. Go to: https://www.nitrc.org/projects/whs-sd-atlas/")
    print("   2. Register/login to NITRC")
    print("   3. Download 'WHS_SD_rat_atlas_v4.0.nii.gz'")
    print("   4. Download 'WHS_SD_rat_atlas_v4.0_labels.csv'")
    print(f"   5. Move files to: {waxholm_dir}/")
    print()
    print("   Alternative: Download from EBRAINS (requires registration)")
    print("   https://ebrains.eu/service/waxholm-space-atlas-of-the-sprague-dawley-rat-brain")
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Download brain atlases")
    parser.add_argument("--all", action="store_true", help="Download all atlases")
    parser.add_argument("--d99", action="store_true", help="Download D99 macaque atlas")
    parser.add_argument("--allen", action="store_true", help="Download Allen CCF mouse atlas")
    parser.add_argument("--waxholm", action="store_true", help="Download Waxholm rat atlas")
    args = parser.parse_args()
    
    # Create atlas directory
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not any([args.all, args.d99, args.allen, args.waxholm]):
        parser.print_help()
        print("\n📊 Current atlas status:")
        
        d99_ok = (ATLAS_DIR / "D99_v2.0_dist" / "D99_atlas_v2.0.nii.gz").exists()
        allen_ok = (ATLAS_DIR / "Allen_CCF_mouse" / "mouse_connectivity_manifest.json").exists()
        waxholm_ok = bool(list((ATLAS_DIR / "Waxholm_rat").glob("*.nii*")) if (ATLAS_DIR / "Waxholm_rat").exists() else [])
        
        print(f"  {'✅' if d99_ok else '❌'} D99 macaque v2.0")
        print(f"  {'✅' if allen_ok else '❌'} Allen CCF mouse v3")
        print(f"  {'✅' if waxholm_ok else '❌'} Waxholm rat v4")
        return
    
    results = []
    
    if args.all or args.d99:
        results.append(("D99", download_d99()))
    
    if args.all or args.allen:
        results.append(("Allen CCF", download_allen()))
    
    if args.all or args.waxholm:
        results.append(("Waxholm", download_waxholm()))
    
    print("\n📊 Summary:")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")


if __name__ == "__main__":
    main()
