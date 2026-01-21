#!/bin/bash
# TwoSphere MCP Services Startup Script
# Starts twosphere-mcp (8006), brain-atlas (8007), prime-de (8009) servers
#
# Usage:
#   ./scripts/start_twosphere_services.sh           # Start services
#   ./scripts/start_twosphere_services.sh --stop    # Stop services
#   ./scripts/start_twosphere_services.sh --status  # Check status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/twosphere.log"

# Conda environment to use
CONDA_ENV="twosphere"

# Ports
TWOSPHERE_PORT=8006
BRAIN_ATLAS_PORT=8007
PRIME_DE_PORT=8009

# Parse arguments
ACTION="start"
for arg in "$@"; do
    case $arg in
        --stop)
            ACTION="stop"
            ;;
        --status)
            ACTION="status"
            ;;
        --restart)
            ACTION="restart"
            ;;
    esac
done

# Functions
stop_services() {
    echo "🛑 Stopping twosphere services..."
    pgrep -f "twosphere_http_server.py" | xargs kill -9 2>/dev/null
    pgrep -f "brain_atlas_http_server.py" | xargs kill -9 2>/dev/null
    pgrep -f "prime_de_http_server.py" | xargs kill -9 2>/dev/null
    sleep 1
    
    for port in $TWOSPHERE_PORT $BRAIN_ATLAS_PORT $PRIME_DE_PORT; do
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "⚠️  Port $port still in use, force killing..."
            lsof -ti:$port | xargs kill -9 2>/dev/null
        fi
    done
    
    echo "✅ Services stopped"
}

check_status() {
    echo "📊 TwoSphere MCP Service Status"
    echo "================================"
    
    if lsof -ti:$TWOSPHERE_PORT >/dev/null 2>&1; then
        echo "✅ twosphere-mcp:  running on port $TWOSPHERE_PORT"
        curl -s http://localhost:$TWOSPHERE_PORT/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "   (health check failed)"
    else
        echo "❌ twosphere-mcp:  not running (port $TWOSPHERE_PORT)"
    fi
    
    echo ""
    
    if lsof -ti:$BRAIN_ATLAS_PORT >/dev/null 2>&1; then
        echo "✅ brain-atlas:    running on port $BRAIN_ATLAS_PORT"
        curl -s http://localhost:$BRAIN_ATLAS_PORT/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "   (health check failed)"
    else
        echo "❌ brain-atlas:    not running (port $BRAIN_ATLAS_PORT)"
    fi
    
    echo ""
    
    if lsof -ti:$PRIME_DE_PORT >/dev/null 2>&1; then
        echo "✅ prime-de:       running on port $PRIME_DE_PORT"
        curl -s http://localhost:$PRIME_DE_PORT/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "   (health check failed)"
    else
        echo "❌ prime-de:       not running (port $PRIME_DE_PORT)"
    fi
}

start_services() {
    echo "🧠 Starting TwoSphere MCP Services"
    echo "==================================="
    echo "📁 Project: $PROJECT_ROOT"
    echo "🐍 Conda env: $CONDA_ENV"
    echo "📝 Log file: $LOG_FILE"
    echo ""
    
    # Activate conda environment
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    
    cd "$PROJECT_ROOT"
    
    # Start twosphere-mcp (port 8006)
    if ! lsof -ti:$TWOSPHERE_PORT >/dev/null 2>&1; then
        echo "🚀 Starting twosphere-mcp on port $TWOSPHERE_PORT..."
        nohup python3 bin/twosphere_http_server.py --port $TWOSPHERE_PORT >> "$LOG_FILE" 2>&1 &
        TWOSPHERE_PID=$!
        echo "   PID: $TWOSPHERE_PID"
    else
        echo "✅ twosphere-mcp already running on port $TWOSPHERE_PORT"
    fi
    
    # Start brain-atlas (port 8007)
    if ! lsof -ti:$BRAIN_ATLAS_PORT >/dev/null 2>&1; then
        echo "🚀 Starting brain-atlas on port $BRAIN_ATLAS_PORT..."
        nohup python3 bin/brain_atlas_http_server.py --port $BRAIN_ATLAS_PORT >> "$LOG_FILE" 2>&1 &
        ATLAS_PID=$!
        echo "   PID: $ATLAS_PID"
    else
        echo "✅ brain-atlas already running on port $BRAIN_ATLAS_PORT"
    fi
    
    # Start prime-de (port 8009)
    if ! lsof -ti:$PRIME_DE_PORT >/dev/null 2>&1; then
        echo "🚀 Starting prime-de on port $PRIME_DE_PORT..."
        nohup python3 bin/prime_de_http_server.py --port $PRIME_DE_PORT >> "$LOG_FILE" 2>&1 &
        PRIME_PID=$!
        echo "   PID: $PRIME_PID"
    else
        echo "✅ prime-de already running on port $PRIME_DE_PORT"
    fi
    
    # Wait for startup
    echo ""
    echo "⏳ Waiting for services to start..."
    sleep 3
    
    # Verify
    echo ""
    check_status
    
    echo ""
    echo "📋 Monitor logs: tail -f $LOG_FILE"
}

# Main
case $ACTION in
    stop)
        stop_services
        ;;
    status)
        check_status
        ;;
    restart)
        stop_services
        sleep 1
        start_services
        ;;
    start)
        start_services
        ;;
esac
