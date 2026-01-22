"""Ernie2 Swarm integration for domain-expert query augmentation.

Provides access to merge2docs' 36 domain-specific collections including:
- docs_library_neuroscience_MRI (brain communication, fMRI)
- docs_library_bioengineering_LOC (Lab-on-Chip, biosensing)
- docs_library_physics_optics (photonics, spectroscopy)
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any


class Ernie2SwarmClient:
    """Client for querying merge2docs ernie2_swarm collections."""

    def __init__(
        self,
        ernie2_path: Optional[str] = None,
        use_cloud: bool = False
    ):
        """
        Initialize Ernie2 swarm client.

        Args:
            ernie2_path: Path to ernie2_swarm.py script
                        (default: ../merge2docs/bin/ernie2_swarm.py)
            use_cloud: If True, use --cloud (Groq), else --local (MLX)
        """
        if ernie2_path is None:
            # Default to merge2docs sibling directory
            self.ernie2_path = Path(__file__).parent.parent.parent.parent.parent / \
                              "merge2docs" / "bin" / "ernie2_swarm.py"
        else:
            self.ernie2_path = Path(ernie2_path)

        if not self.ernie2_path.exists():
            raise FileNotFoundError(
                f"ernie2_swarm.py not found at {self.ernie2_path}\n"
                "Please ensure merge2docs is cloned as ../merge2docs"
            )

        self.use_cloud = use_cloud

    async def query(
        self,
        question: str,
        collections: Optional[List[str]] = None,
        expert_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query ernie2_swarm collections for domain-expert insights.

        Args:
            question: Question to ask domain experts
            collections: List of collection names to query
                        (e.g., ["docs_library_neuroscience_MRI"])
            expert_agents: Optional list of specific domain experts

        Returns:
            Dict with keys:
                - answer: Expert response text
                - collections_queried: List of collections used
                - parameters: Extracted parameter suggestions (if applicable)

        Example:
            >>> client = Ernie2SwarmClient()
            >>> result = await client.query(
            ...     question="What refractive index sensitivity for GABA detection?",
            ...     collections=["neuroscience_MRI", "physics_optics"]
            ... )
            >>> print(result['parameters']['sensitivity'])
        """
        def _run_query():
            # Build command
            cmd = ["python", str(self.ernie2_path)]

            # Add collection filter
            if collections:
                collections_str = ",".join(collections)
                cmd.extend(["--collection", collections_str])

            # Add question
            cmd.extend(["--question", question])

            # Add model selection
            if self.use_cloud:
                cmd.append("--cloud")
            else:
                cmd.append("--local")

            # Run command
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True
                )

                # Parse output (assuming JSON or text)
                output = result.stdout.strip()

                # Try to parse as JSON
                try:
                    parsed = json.loads(output)
                    return {
                        "answer": parsed.get("answer", output),
                        "collections_queried": collections or [],
                        "parameters": self._extract_parameters(parsed),
                        "raw_response": parsed
                    }
                except json.JSONDecodeError:
                    # Plain text response
                    return {
                        "answer": output,
                        "collections_queried": collections or [],
                        "parameters": self._extract_parameters_from_text(output),
                        "raw_response": output
                    }

            except subprocess.TimeoutExpired:
                return {
                    "answer": "Query timeout",
                    "collections_queried": collections or [],
                    "parameters": {},
                    "error": "Timeout after 30s"
                }
            except subprocess.CalledProcessError as e:
                return {
                    "answer": f"Query failed: {e.stderr}",
                    "collections_queried": collections or [],
                    "parameters": {},
                    "error": str(e)
                }

        return await asyncio.to_thread(_run_query)

    def _extract_parameters(self, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameter suggestions from JSON response."""
        params = {}

        # Look for common parameter keys
        if isinstance(parsed_response, dict):
            # Refractive index sensitivity
            if "refractive_index_sensitivity" in parsed_response:
                params["refractive_index_sensitivity"] = \
                    parsed_response["refractive_index_sensitivity"]

            # Wavelength
            if "wavelength_nm" in parsed_response:
                params["wavelength_nm"] = parsed_response["wavelength_nm"]

            # Path length
            if "path_length_mm" in parsed_response:
                params["path_length_mm"] = parsed_response["path_length_mm"]

            # Frequency bands (for MRI)
            if "frequency_bands" in parsed_response:
                params["frequency_bands"] = parsed_response["frequency_bands"]

        return params

    def _extract_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """Extract numerical parameters from text response."""
        params = {}

        # Simple regex-based extraction
        import re

        # Look for patterns like "sensitivity: 1e-6", "sensitivity should be around 1e-6"
        # Match scientific notation near "sensitivity" keyword
        sensitivity_match = re.search(
            r'sensitivity.*?([0-9.]+e[+-]?[0-9]+)',
            text,
            re.IGNORECASE
        )
        if sensitivity_match:
            params["refractive_index_sensitivity"] = float(sensitivity_match.group(1))

        # Look for wavelength (e.g., "633 nm", "850nm")
        wavelength_match = re.search(
            r'(\d+)\s*nm',
            text
        )
        if wavelength_match:
            params["wavelength_nm"] = float(wavelength_match.group(1))

        return params


class YadaServicesMCPClient:
    """Client for querying ernie2_swarm via yada-services-secure MCP server.

    Uses the REST /api/call_tool endpoint on port 8002.
    """

    def __init__(self, host: str = "localhost", port: int = 8002):
        """Initialize MCP client.

        Args:
            host: yada-services-secure host
            port: yada-services-secure port (default: 8002)
        """
        self.base_url = f"http://{host}:{port}"

    async def health_check(self) -> bool:
        """Check if yada-services-secure is running."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via REST endpoint.

        Args:
            tool_name: Name of the tool (e.g., "ernie2_swarm")
            arguments: Tool arguments

        Returns:
            Tool result or error dict
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "name": tool_name,
                    "arguments": arguments
                }
                async with session.post(
                    f"{self.base_url}/api/call_tool",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    result = await resp.json()
                    return result
        except ImportError:
            # Fallback to requests if aiohttp not available
            import requests
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            resp = requests.post(
                f"{self.base_url}/api/call_tool",
                json=payload,
                timeout=120
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def query_ernie2(
        self,
        question: str,
        context: Optional[str] = None,
        collections: Optional[List[str]] = None,
        max_steps: int = 3,
        style: str = "detailed"
    ) -> Dict[str, Any]:
        """Query ernie2_swarm via yada-services-secure.

        Args:
            question: Question to ask domain experts
            context: Optional context for the query
            collections: Optional list of collections (comma-separated string)
            max_steps: Number of minion steps
            style: Response style ("concise" or "detailed")

        Returns:
            Dict with answer or error
        """
        arguments = {
            "query": question,
            "max_steps": max_steps,
            "style": style
        }
        if context:
            arguments["context"] = context
        if collections:
            arguments["collections"] = ",".join(collections)

        result = await self.call_tool("ernie2_swarm", arguments)

        # Handle result format
        if "result" in result:
            return {
                "answer": result["result"],
                "collections_queried": collections or [],
                "success": True
            }
        elif "error" in result:
            return {
                "answer": None,
                "error": result["error"],
                "collections_queried": collections or [],
                "success": False
            }
        else:
            return {
                "answer": str(result),
                "collections_queried": collections or [],
                "success": True
            }


# Convenience function for MCP tools
async def query_expert_collections(
    question: str,
    collections: List[str],
    use_cloud: bool = False,
    use_mcp: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for querying ernie2_swarm from MCP tools.

    Args:
        question: Question to ask
        collections: List of collection names
        use_cloud: Use Groq (cloud) or MLX (local) - only for subprocess mode
        use_mcp: If True, use yada-services-secure MCP (recommended)
                 If False, use subprocess to call ernie2_swarm.py directly

    Returns:
        Query result dict

    Example:
        >>> result = await query_expert_collections(
        ...     question="What sensitivity for neurotransmitter detection?",
        ...     collections=["neuroscience_MRI", "bioengineering_LOC"]
        ... )
    """
    if use_mcp:
        client = YadaServicesMCPClient()
        return await client.query_ernie2(question, collections=collections)
    else:
        client = Ernie2SwarmClient(use_cloud=use_cloud)
        return await client.query(question, collections)
