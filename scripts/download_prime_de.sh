#!/bin/bash
# Download PRIME-DE (PRIMatE Data Exchange) neuroimaging data
# Data is stored in ~/data/prime_de (outside repo to avoid git)
#
# Usage:
#   ./scripts/download_prime_de.sh                    # Download all available
#   ./scripts/download_prime_de.sh --list             # List available datasets
#   ./scripts/download_prime_de.sh --dataset bordeaux # Download specific dataset
#   ./scripts/download_prime_de.sh --subjects 3       # Limit subjects per dataset

set -e

# Configuration
DATA_DIR="${PRIME_DE_DATA_DIR:-$HOME/data/prime_de}"
S3_BASE="https://fcp-indi.s3.amazonaws.com"
S3_PREFIX="data/Projects/INDI/PRIME"

# Available datasets (publicly accessible on S3)
DATASETS=(
    "BORDEAUX24"
    "Foveola2024NNS"
    "UWMadison"
)

# Parse arguments
ACTION="download"
SPECIFIC_DATASET=""
MAX_SUBJECTS=0  # 0 = all

while [[ $# -gt 0 ]]; do
    case $1 in
        --list)
            ACTION="list"
            shift
            ;;
        --dataset)
            SPECIFIC_DATASET="$2"
            shift 2
            ;;
        --subjects)
            MAX_SUBJECTS="$2"
            shift 2
            ;;
        --help|-h)
            echo "PRIME-DE Data Downloader"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --list              List available datasets"
            echo "  --dataset NAME      Download specific dataset (bordeaux, foveola, uwmadison)"
            echo "  --subjects N        Limit to N subjects per dataset (default: all)"
            echo "  --help              Show this help"
            echo ""
            echo "Data will be stored in: $DATA_DIR"
            echo ""
            echo "Available datasets:"
            echo "  BORDEAUX24      - Macaque 7T MRI (BIDS format, ~8 subjects)"
            echo "  Foveola2024NNS  - Foveal imaging data"
            echo "  UWMadison       - Macaque MRI (30 parts, ~100GB total)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create data directory
mkdir -p "$DATA_DIR"

list_datasets() {
    echo "📋 Available PRIME-DE datasets on S3:"
    echo ""
    for ds in "${DATASETS[@]}"; do
        # Get file count
        count=$(curl -s "${S3_BASE}/?prefix=${S3_PREFIX}/${ds}/&max-keys=1000" | grep -c '<Key>' || echo "?")
        echo "  $ds ($count files)"
    done
    echo ""
    echo "📁 Local data directory: $DATA_DIR"
    if [[ -d "$DATA_DIR" ]]; then
        echo "   Contents:"
        ls -1 "$DATA_DIR" 2>/dev/null | sed 's/^/     /'
    fi
}

download_bids_dataset() {
    local dataset=$1
    local target_dir="$DATA_DIR/$dataset"
    
    echo "📥 Downloading $dataset (BIDS format)..."
    mkdir -p "$target_dir"
    
    # Get dataset description first
    echo "   Fetching metadata..."
    curl -sf -o "$target_dir/dataset_description.json" \
        "${S3_BASE}/${S3_PREFIX}/${dataset}/dataset_description.json" 2>/dev/null || true
    curl -sf -o "$target_dir/README" \
        "${S3_BASE}/${S3_PREFIX}/${dataset}/README" 2>/dev/null || true
    curl -sf -o "$target_dir/CHANGES" \
        "${S3_BASE}/${S3_PREFIX}/${dataset}/CHANGES" 2>/dev/null || true
    curl -sf -o "$target_dir/participants.tsv" \
        "${S3_BASE}/${S3_PREFIX}/${dataset}/participants.tsv" 2>/dev/null || true
    
    # Get list of subjects
    echo "   Finding subjects..."
    subjects=$(curl -s "${S3_BASE}/?delimiter=/&prefix=${S3_PREFIX}/${dataset}/sub-" | \
        grep -oE '<Prefix>[^<]+</Prefix>' | \
        sed 's/<[^>]*>//g' | \
        sed "s|${S3_PREFIX}/${dataset}/||" | \
        sed 's|/$||' | \
        sort -u)
    
    subject_count=$(echo "$subjects" | grep -c 'sub-' || echo 0)
    echo "   Found $subject_count subjects"
    
    # Download each subject
    downloaded=0
    for subj in $subjects; do
        if [[ $MAX_SUBJECTS -gt 0 && $downloaded -ge $MAX_SUBJECTS ]]; then
            echo "   Reached max subjects ($MAX_SUBJECTS), stopping"
            break
        fi
        
        echo "   📦 $subj..."
        
        # Get all files for this subject
        files=$(curl -s "${S3_BASE}/?prefix=${S3_PREFIX}/${dataset}/${subj}/&max-keys=500" | \
            grep -oE '<Key>[^<]+</Key>' | \
            sed 's/<[^>]*>//g')
        
        for file in $files; do
            # Extract relative path
            rel_path=$(echo "$file" | sed "s|${S3_PREFIX}/${dataset}/||")
            local_path="$target_dir/$rel_path"
            
            # Create directory and download
            mkdir -p "$(dirname "$local_path")"
            if [[ ! -f "$local_path" ]]; then
                curl -sf -o "$local_path" "${S3_BASE}/$file" || echo "      Failed: $rel_path"
            fi
        done
        
        ((downloaded++))
    done
    
    # Download derivatives if present
    echo "   Checking derivatives..."
    deriv_files=$(curl -s "${S3_BASE}/?prefix=${S3_PREFIX}/${dataset}/derivatives/&max-keys=500" | \
        grep -oE '<Key>[^<]+</Key>' | \
        sed 's/<[^>]*>//g' || true)
    
    for file in $deriv_files; do
        rel_path=$(echo "$file" | sed "s|${S3_PREFIX}/${dataset}/||")
        local_path="$target_dir/$rel_path"
        mkdir -p "$(dirname "$local_path")"
        if [[ ! -f "$local_path" ]]; then
            curl -sf -o "$local_path" "${S3_BASE}/$file" 2>/dev/null || true
        fi
    done
    
    echo "✅ $dataset downloaded to $target_dir"
}

download_tarball_dataset() {
    local dataset=$1
    local target_dir="$DATA_DIR/$dataset"
    
    echo "📥 Downloading $dataset (tarball format)..."
    mkdir -p "$target_dir"
    
    # List available parts
    parts=$(curl -s "${S3_BASE}/?prefix=${S3_PREFIX}/${dataset}/&max-keys=100" | \
        grep -oE '<Key>[^<]+\.tar\.gz</Key>' | \
        sed 's/<[^>]*>//g')
    
    part_count=$(echo "$parts" | grep -c 'tar.gz' || echo 0)
    echo "   Found $part_count parts"
    
    # Download each part
    for part in $parts; do
        filename=$(basename "$part")
        local_path="$target_dir/$filename"
        
        if [[ -f "$local_path" ]]; then
            echo "   ⏭️  $filename (exists)"
            continue
        fi
        
        echo "   ⬇️  $filename..."
        curl -# -o "$local_path" "${S3_BASE}/$part"
    done
    
    echo "✅ $dataset downloaded to $target_dir"
    echo "   Extract with: cd $target_dir && for f in *.tar.gz; do tar -xzf \$f; done"
}

download_dataset() {
    local dataset=$1
    
    case $dataset in
        BORDEAUX24|bordeaux|bordeaux24)
            download_bids_dataset "BORDEAUX24"
            ;;
        Foveola2024NNS|foveola|foveola2024)
            download_bids_dataset "Foveola2024NNS"
            ;;
        UWMadison|uwmadison|wisconsin)
            download_tarball_dataset "UWMadison"
            ;;
        *)
            echo "❌ Unknown dataset: $dataset"
            echo "   Available: bordeaux, foveola, uwmadison"
            exit 1
            ;;
    esac
}

# Main
case $ACTION in
    list)
        list_datasets
        ;;
    download)
        if [[ -n "$SPECIFIC_DATASET" ]]; then
            download_dataset "$SPECIFIC_DATASET"
        else
            echo "🧠 PRIME-DE Data Downloader"
            echo "==========================="
            echo "Data directory: $DATA_DIR"
            echo ""
            
            # Download BIDS datasets (smaller, more useful)
            download_bids_dataset "BORDEAUX24"
            echo ""
            
            # Skip large tarball datasets by default
            echo "ℹ️  UWMadison dataset (~100GB) skipped by default"
            echo "   To download: $0 --dataset uwmadison"
        fi
        ;;
esac

echo ""
echo "📁 Data location: $DATA_DIR"
echo "🔗 Symlink to repo: ln -s $DATA_DIR /path/to/twosphere-mcp/data/prime_de"
