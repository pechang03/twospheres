"""Client for merge2docs tensor routing system.

Provides query routing to appropriate domain-specific cells and tools.
The tensor system organizes knowledge/tools in a 3D structure: [LLM, FI_Level, Domain]
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
import aiohttp
import logging

logger = logging.getLogger(__name__)


class TensorRoutingClient:
    """Client for tensor routing API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None
    ):
        """Initialize tensor routing client.

        Args:
            base_url: Base URL (default: http://localhost)
            port: Port number (default: from TENSOR_ROUTING_PORT env or 8091)
        """
        self.base_url = base_url or os.getenv("TENSOR_ROUTING_URL", "http://localhost")
        self.port = port or int(os.getenv("TENSOR_ROUTING_PORT", "8091"))
        self.endpoint = f"{self.base_url}:{self.port}"
        logger.info(f"Tensor routing endpoint: {self.endpoint}")

    async def route_query(
        self,
        query: str,
        domain_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Route a query to appropriate domain cell.

        Args:
            query: Query string to route
            domain_hint: Optional domain hint (e.g., 'math', 'physics', 'bio')

        Returns:
            Dict with routing information:
                - cell_position: [llm_idx, fi_idx, domain_idx]
                - domain: Domain name
                - fi_level: FI level (F0-F6)
                - fi_name: FI level name
                - tools: List of recommended tools
                - routing_info: Additional routing metadata
                - engram_hit_rate: Cache hit rate

        Example:
            >>> client = TensorRoutingClient()
            >>> result = await client.route_query("prove the intermediate value theorem")
            >>> print(result['domain'])  # 'math'
            >>> print(result['tools'])   # ['paper_search', 'theorem_match', ...]
        """
        try:
            params = {"query": query}
            if domain_hint:
                params["domain"] = domain_hint

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}/tensor/route",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Tensor routing failed: {response.status}")
                        return self._fallback_routing(query, domain_hint)

        except aiohttp.ClientError as e:
            logger.error(f"Tensor routing connection error: {e}")
            return self._fallback_routing(query, domain_hint)
        except Exception as e:
            logger.error(f"Tensor routing error: {e}")
            return self._fallback_routing(query, domain_hint)

    def _fallback_routing(
        self,
        query: str,
        domain_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback routing when tensor service unavailable.

        Provides basic routing based on keyword matching.
        """
        query_lower = query.lower()

        # Simple keyword-based domain detection
        if domain_hint:
            domain = domain_hint
        elif any(kw in query_lower for kw in ['brain', 'mri', 'fmri', 'neuron', 'cortex']):
            domain = 'neuroscience'
        elif any(kw in query_lower for kw in ['optics', 'photon', 'laser', 'fiber', 'resonator']):
            domain = 'physics'
        elif any(kw in query_lower for kw in ['drug', 'biomarker', 'organ', 'chip', 'microfluidic']):
            domain = 'bioengineering'
        elif any(kw in query_lower for kw in ['optimize', 'minimize', 'maximize', 'convergence']):
            domain = 'mathematics'
        else:
            domain = 'general'

        # Map to FI level based on query complexity
        if any(kw in query_lower for kw in ['system', 'integrate', 'combine', 'overall']):
            fi_level = 'F4'
            fi_name = 'F4_INTEGRATION'
        elif any(kw in query_lower for kw in ['design', 'plan', 'strategy', 'optimize']):
            fi_level = 'F5'
            fi_name = 'F5_PLANNING'
        else:
            fi_level = 'F2'
            fi_name = 'F2_COMPOSITION'

        return {
            'cell_position': [0, self._fi_to_index(fi_level), self._domain_to_index(domain)],
            'domain': domain,
            'fi_level': fi_level,
            'fi_name': fi_name,
            'tools': self._get_default_tools(domain),
            'routing_info': {
                'cell_address': f'Tensor[LLM, {fi_level}, {domain}]',
                'fallback': True
            },
            'engram_hit_rate': 0.0
        }

    def _fi_to_index(self, fi_level: str) -> int:
        """Convert FI level string to index."""
        mapping = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4, 'F5': 5, 'F6': 6}
        return mapping.get(fi_level, 2)

    def _domain_to_index(self, domain: str) -> int:
        """Convert domain string to index."""
        domains = ['math', 'physics', 'neuroscience', 'bioengineering', 'general']
        try:
            return domains.index(domain)
        except ValueError:
            return 4  # general

    def _get_default_tools(self, domain: str) -> List[str]:
        """Get default tools for domain."""
        tool_map = {
            'neuroscience': ['fmri_analysis', 'network_analysis', 'connectivity_metrics'],
            'physics': ['ray_tracing', 'wavefront_analysis', 'optical_design'],
            'bioengineering': ['loc_simulation', 'microfluidic_analysis', 'biomarker_detection'],
            'mathematics': ['optimization', 'numerical_analysis', 'statistical_analysis'],
            'general': ['search', 'analysis', 'synthesis']
        }
        return tool_map.get(domain, tool_map['general'])

    async def get_tools_for_domain(
        self,
        domain: str,
        fi_level: Optional[str] = None
    ) -> List[str]:
        """Get recommended tools for specific domain and FI level.

        Args:
            domain: Domain name
            fi_level: Optional FI level (F0-F6)

        Returns:
            List of tool names
        """
        try:
            params = {"domain": domain}
            if fi_level:
                params["fi_level"] = fi_level

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.endpoint}/tensor/tools",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('tools', [])
                    else:
                        return self._get_default_tools(domain)

        except Exception as e:
            logger.error(f"Failed to get tools: {e}")
            return self._get_default_tools(domain)


# Convenience function
async def route_query(query: str, domain_hint: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to route a query.

    Args:
        query: Query string
        domain_hint: Optional domain hint

    Returns:
        Routing result dict

    Example:
        >>> result = await route_query("optimize fiber coupling efficiency")
        >>> print(result['domain'])  # 'physics'
    """
    client = TensorRoutingClient()
    return await client.route_query(query, domain_hint)


__all__ = ['TensorRoutingClient', 'route_query']
