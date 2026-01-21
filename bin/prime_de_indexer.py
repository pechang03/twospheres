#!/usr/bin/env python3
"""PRIME-DE Database Indexer - Index PRIME-DE datasets into PostgreSQL.

Scans PRIME-DE data directory and indexes all datasets/subjects into
the twosphere_brain database for fast lookups.

Usage:
    python bin/prime_de_indexer.py --scan
    python bin/prime_de_indexer.py --scan --dataset BORDEAUX24
    python bin/prime_de_indexer.py --verify

This eliminates filesystem scanning in prime_de_http_server.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Try to import nibabel (optional - only needed for NIfTI metadata extraction)
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    nib = None
    HAS_NIBABEL = False

# Configuration
DATA_DIR = Path(os.getenv("PRIME_DE_DATA_DIR", Path.home() / "data" / "prime_de"))
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5432
POSTGRES_USER = "petershaw"
POSTGRES_PASSWORD = "FruitSalid4"
DATABASE = "twosphere_brain"


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=DATABASE,
        cursor_factory=RealDictCursor
    )


def scan_subject(dataset_path: Path, subject_dir: Path) -> Optional[Dict]:
    """Scan a single subject and extract metadata."""
    if not subject_dir.is_dir():
        return None

    subject_name = subject_dir.name
    if not subject_name.startswith("sub-"):
        return None

    # Remove "sub-" prefix for subject_name
    subject_name = subject_name[4:]

    # Find all NIfTI files
    nifti_files = []
    for nii_file in subject_dir.rglob("*.nii*"):
        rel_path = str(nii_file.relative_to(subject_dir))

        # Determine modality and suffix
        modality = "unknown"
        suffix = "unknown"

        if "/anat/" in str(nii_file) or rel_path.startswith("anat/"):
            modality = "anat"
            if "T1w" in nii_file.name:
                suffix = "T1w"
            elif "T2w" in nii_file.name:
                suffix = "T2w"
            elif "FLAIR" in nii_file.name:
                suffix = "FLAIR"
        elif "/func/" in str(nii_file) or rel_path.startswith("func/"):
            modality = "func"
            if "bold" in nii_file.name:
                suffix = "bold"
        elif "/dwi/" in str(nii_file) or rel_path.startswith("dwi/"):
            modality = "dwi"
            suffix = "dwi"

        # Try to get NIfTI metadata (timepoints, TR) - only if nibabel available
        timepoints = None
        tr = None
        if HAS_NIBABEL:
            try:
                img = nib.load(str(nii_file))
                if len(img.shape) == 4:  # 4D image (timeseries)
                    timepoints = img.shape[3]

                # Try to get TR from header
                if hasattr(img.header, 'get_zooms') and len(img.header.get_zooms()) > 3:
                    tr = float(img.header.get_zooms()[3])
            except Exception as e:
                # If nibabel fails, continue without metadata
                print(f"  ⚠️  Could not read NIfTI metadata for {nii_file.name}: {e}")

        nifti_files.append({
            "path": str(nii_file),
            "relative_path": rel_path,
            "modality": modality,
            "suffix": suffix,
            "timepoints": timepoints,
            "tr": tr,
        })

    if not nifti_files:
        return None

    return {
        "dataset_name": dataset_path.name,
        "subject_name": subject_name,
        "nifti_files": nifti_files,
    }


def index_dataset(dataset_name: Optional[str] = None) -> Dict[str, int]:
    """Index datasets into PostgreSQL.

    Args:
        dataset_name: If provided, only index this dataset. Otherwise, index all.

    Returns:
        Statistics dict with counts.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    stats = {
        "datasets_scanned": 0,
        "subjects_indexed": 0,
        "files_indexed": 0,
        "subjects_updated": 0,
    }

    try:
        # Get datasets to scan
        if dataset_name:
            dataset_paths = [DATA_DIR / dataset_name]
        else:
            dataset_paths = [d for d in DATA_DIR.iterdir() if d.is_dir()]

        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                continue

            print(f"\n📁 Scanning dataset: {dataset_path.name}")
            stats["datasets_scanned"] += 1

            # Find all subjects
            subjects = list(dataset_path.glob("sub-*"))
            print(f"   Found {len(subjects)} subjects")

            for subject_dir in subjects:
                # Scan subject
                subject_data = scan_subject(dataset_path, subject_dir)
                if not subject_data:
                    continue

                dataset = subject_data["dataset_name"]
                subject = subject_data["subject_name"]
                nifti_files = subject_data["nifti_files"]

                # Get primary NIfTI file (prefer T1w, then bold, then first available)
                primary_file = None
                primary_timepoints = None
                primary_tr = None

                for nf in nifti_files:
                    if nf["suffix"] == "T1w":
                        primary_file = nf["path"]
                        primary_timepoints = nf["timepoints"]
                        primary_tr = nf["tr"]
                        break

                if not primary_file:
                    for nf in nifti_files:
                        if nf["suffix"] == "bold":
                            primary_file = nf["path"]
                            primary_timepoints = nf["timepoints"]
                            primary_tr = nf["tr"]
                            break

                if not primary_file and nifti_files:
                    nf = nifti_files[0]
                    primary_file = nf["path"]
                    primary_timepoints = nf["timepoints"]
                    primary_tr = nf["tr"]

                # Check if subject already exists
                cursor.execute("""
                    SELECT subject_id FROM prime_de_subjects
                    WHERE dataset_name = %s AND subject_name = %s
                """, (dataset, subject))

                existing = cursor.fetchone()

                if existing:
                    # Update existing
                    cursor.execute("""
                        UPDATE prime_de_subjects
                        SET nifti_path = %s,
                            timepoints = %s,
                            tr = %s
                        WHERE dataset_name = %s AND subject_name = %s
                    """, (primary_file, primary_timepoints, primary_tr, dataset, subject))
                    stats["subjects_updated"] += 1
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO prime_de_subjects
                        (dataset_name, subject_name, nifti_path, timepoints, tr, processed)
                        VALUES (%s, %s, %s, %s, %s, FALSE)
                    """, (dataset, subject, primary_file, primary_timepoints, primary_tr))
                    stats["subjects_indexed"] += 1

                stats["files_indexed"] += len(nifti_files)
                print(f"   ✅ {subject}: {len(nifti_files)} files (primary: {Path(primary_file).name if primary_file else 'none'})")

        # Commit changes
        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error during indexing: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

    return stats


def verify_index() -> Dict[str, int]:
    """Verify indexed data in database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Count subjects by dataset
        cursor.execute("""
            SELECT dataset_name, COUNT(*) as count
            FROM prime_de_subjects
            GROUP BY dataset_name
            ORDER BY dataset_name
        """)

        datasets = cursor.fetchall()

        print("\n📊 Indexed PRIME-DE Subjects:")
        total = 0
        for row in datasets:
            count = row["count"]
            total += count
            print(f"   {row['dataset_name']}: {count} subjects")

        print(f"\n   Total: {total} subjects")

        # Check for subjects with timepoints (4D data)
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM prime_de_subjects
            WHERE timepoints IS NOT NULL AND timepoints > 1
        """)

        timeseries_count = cursor.fetchone()["count"]
        print(f"   Timeseries (4D): {timeseries_count} subjects")

        return {
            "total_subjects": total,
            "datasets": len(datasets),
            "timeseries_subjects": timeseries_count,
        }

    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Index PRIME-DE datasets into PostgreSQL"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and index datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Only scan specific dataset"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify indexed data"
    )

    args = parser.parse_args()

    if not any([args.scan, args.verify]):
        parser.print_help()
        return

    print("🧠 PRIME-DE Database Indexer")
    print(f"Data directory: {DATA_DIR}")
    print(f"Database: {DATABASE}")

    if args.scan:
        print("\n🔍 Scanning datasets...")
        stats = index_dataset(args.dataset)

        print("\n✅ Indexing complete!")
        print(f"   Datasets scanned: {stats['datasets_scanned']}")
        print(f"   Subjects indexed: {stats['subjects_indexed']}")
        print(f"   Subjects updated: {stats['subjects_updated']}")
        print(f"   Files found: {stats['files_indexed']}")

    if args.verify:
        verify_index()


if __name__ == "__main__":
    main()
