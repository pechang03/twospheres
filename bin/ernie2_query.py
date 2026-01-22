#!/usr/bin/env python3
"""Simple ernie2_swarm wrapper with MCP fallback.

Tries yada-services-secure first via ernie2_swarm_mcp_e.py.
Falls back to direct ernie2_swarm.py execution if MCP fails.

Usage:
    python bin/ernie2_query.py "What is disc dimension?"
    python bin/ernie2_query.py "Explain glymphatic system" --collections neuroscience_MRI
"""

import json
import os
import subprocess
import sys
from pathlib import Path


# Paths to the two ernie2 scripts
MERGE2DOCS = Path(__file__).parent.parent.parent / "merge2docs"
ERNIE2_MCP = MERGE2DOCS / "bin" / "ernie2_swarm_mcp_e.py"
ERNIE2_DIRECT = MERGE2DOCS / "bin" / "ernie2_swarm.py"


def try_mcp_query(question: str, collections: list = None, num_minions: int = 2) -> dict:
    """Try query via yada-services-secure MCP."""
    cmd = [sys.executable, str(ERNIE2_MCP), "--question", question, "--num-minions", str(num_minions)]

    if collections:
        for c in collections:
            cmd.extend(["--collection", c])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(MERGE2DOCS),
        )

        # Check for known server errors
        if "No module named" in result.stderr or "Error:" in result.stdout:
            return {"error": result.stderr or result.stdout, "fallback_needed": True}

        return {
            "success": True,
            "answer": result.stdout.strip(),
            "method": "mcp"
        }
    except subprocess.TimeoutExpired:
        return {"error": "MCP timeout", "fallback_needed": True}
    except Exception as e:
        return {"error": str(e), "fallback_needed": True}


def try_direct_query(question: str, collections: list = None, num_minions: int = 2) -> dict:
    """Fallback: run ernie2_swarm.py directly."""
    cmd = [sys.executable, str(ERNIE2_DIRECT), "--question", question, "--num-minions", str(num_minions)]

    if collections:
        for c in collections:
            cmd.extend(["--collection", c])

    # Add --local for free local inference
    cmd.append("--local")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # Direct execution may take longer
            cwd=str(MERGE2DOCS),
        )

        if result.returncode != 0:
            return {"error": result.stderr, "success": False}

        return {
            "success": True,
            "answer": result.stdout.strip(),
            "method": "direct"
        }
    except subprocess.TimeoutExpired:
        return {"error": "Direct query timeout", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def query_ernie2(question: str, collections: list = None, num_minions: int = 2) -> dict:
    """Query ernie2_swarm with automatic fallback.

    Args:
        question: Question to ask
        collections: Optional list of collection names
        num_minions: Number of minion workers

    Returns:
        dict with 'answer' or 'error'
    """
    # Try MCP first
    result = try_mcp_query(question, collections, num_minions)

    if result.get("fallback_needed"):
        sys.stderr.write(f"[ernie2_query] MCP failed ({result.get('error', 'unknown')}), falling back to direct...\n")
        result = try_direct_query(question, collections, num_minions)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query ernie2_swarm with MCP fallback")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-q", "--question", dest="question_arg", help="Alternative way to specify question")
    parser.add_argument("-c", "--collection", "--collections", action="append", dest="collections",
                        help="Collection names (can specify multiple)")
    parser.add_argument("-n", "--num-minions", type=int, default=2, help="Number of minions")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of plain text")

    args = parser.parse_args()

    question = args.question or args.question_arg
    if not question:
        parser.print_help()
        sys.exit(1)

    result = query_ernie2(question, args.collections, args.num_minions)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("success"):
            print(result.get("answer", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
