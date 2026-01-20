#!/bin/bash
# TwoSphere MCP Setup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔬 Setting up TwoSphere MCP..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "📦 Conda detected"
    
    # Check if twosphere env exists
    if conda env list | grep -q "twosphere"; then
        echo "✓ Conda environment 'twosphere' already exists"
        echo "  Activate with: conda activate twosphere"
    else
        echo "Creating conda environment 'twosphere'..."
        conda create -n twosphere python=3.11 -y
        echo "✓ Environment created. Activate with: conda activate twosphere"
    fi
else
    echo "⚠ Conda not found. Using system Python."
fi

# Create data directory and symlink
echo "📁 Setting up data directory..."
mkdir -p data

if [ -d "$HOME/MRISpheres/twospheres" ]; then
    if [ ! -L "data/twospheres" ]; then
        ln -s "$HOME/MRISpheres/twospheres" data/twospheres
        echo "✓ Linked data/twospheres -> ~/MRISpheres/twospheres"
    else
        echo "✓ data/twospheres symlink already exists"
    fi
else
    echo "⚠ ~/MRISpheres/twospheres not found. Skipping symlink."
fi

# Copy environment file
if [ ! -f ".env_twosphere" ]; then
    cp .env_twosphere.example .env_twosphere
    echo "✓ Created .env_twosphere from example"
else
    echo "✓ .env_twosphere already exists"
fi

# Install dependencies (if in active conda/venv)
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -n "$VIRTUAL_ENV" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ No active environment. Install dependencies manually:"
    echo "   pip install -r requirements.txt"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. conda activate twosphere"
echo "  2. pip install -r requirements.txt"
echo "  3. python bin/twosphere_mcp.py  # stdio MCP server"
echo "  4. python bin/twosphere_http_server.py --port 8006  # HTTP server"
