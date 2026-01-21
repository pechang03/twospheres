# Task: Expose Mathematical Validation Tools in yada-services-secure

**Priority**: LOW (Task 4 from ernie2_swarm query)
**Status**: PENDING
**Assigned**: Human/Claude to coordinate with merge2docs team

## Context

The QEC tensor design uses sophisticated mathematical validation:
- Bipartite graph analysis (requirements → tasks)
- RB-domination (critical path identification)
- Treewidth computation (coupling analysis)
- FPT parameter validation (complexity bounds)

These tools **already exist** in merge2docs at:
- `merge2docs/src/backend/algorithms/bipartite_analysis.py`
- `merge2docs/src/backend/algorithms/rb_domination.py`
- `merge2docs/src/backend/algorithms/treewidth.py`
- `merge2docs/src/backend/algorithms/fpt_validator.py`

**Objective**: Expose these tools as HTTP/MCP endpoints in yada-services-secure for automated design validation.

## Architecture

### Current State

merge2docs has these validation tools as Python modules, but they're not exposed via HTTP/MCP for external use.

### Target State

```
┌─────────────────────┐
│  twosphere-mcp      │
│  Design Review      │
└──────────┬──────────┘
           │
           │ HTTP POST /api/validate/*
           ↓
┌─────────────────────────────┐
│  yada-services-secure       │
│  MCP Server (port 8003)     │
├─────────────────────────────┤
│  HTTP Endpoints:            │
│  - /api/validate/bipartite  │
│  - /api/validate/rb-dom     │
│  - /api/validate/treewidth  │
│  - /api/validate/fpt        │
├─────────────────────────────┤
│  MCP Tools:                 │
│  - validate_bipartite       │
│  - validate_rb_domination   │
│  - validate_treewidth       │
│  - validate_fpt_complexity  │
└──────────┬──────────────────┘
           │
           │ Import from merge2docs
           ↓
┌─────────────────────────────┐
│  merge2docs                 │
│  /src/backend/algorithms/   │
│  - bipartite_analysis.py    │
│  - rb_domination.py         │
│  - treewidth.py             │
│  - fpt_validator.py         │
└─────────────────────────────┘
```

## API Specification

### 1. Bipartite Graph Validation

**Endpoint**: `POST /api/validate/bipartite`

**Request**:
```json
{
  "requirements": [
    "BEAD-QEC-1: Bootstrap from merge2docs tensor",
    "BEAD-QEC-2: Function functor from PRIME-DE",
    "BEAD-QEC-3: r-IDS connectivity graph",
    "BEAD-QEC-4: Merge2docs endpoints for corpus",
    "BEAD-QEC-5: Cross-training patterns",
    "BEAD-QEC-6: PRIME-DE integration",
    "BEAD-QEC-7: LRU cache",
    "BEAD-QEC-8: Health validation",
    "BEAD-QEC-9: Documentation"
  ],
  "tasks": [
    "bootstrap_brain_tensor()",
    "PRIMEDELoader.load_subject()",
    "compute_rids_connections()",
    "download_corpus()",
    "apply_cross_training()",
    "BrainRegionCache.get_or_load()",
    "validate_design()"
  ]
}
```

**Response**:
```json
{
  "bipartite_graph": {
    "left_nodes": 9,
    "right_nodes": 7,
    "edges": 15,
    "coverage": 0.667,
    "unmapped_requirements": [
      "BEAD-QEC-4",
      "BEAD-QEC-8",
      "BEAD-QEC-9"
    ]
  },
  "health_impact": 0.05
}
```

### 2. RB-Domination (Critical Path)

**Endpoint**: `POST /api/validate/rb-domination`

**Request**:
```json
{
  "graph": {
    "nodes": ["BEAD-QEC-1", "BEAD-QEC-2", ..., "BEAD-QEC-9"],
    "edges": [
      ["BEAD-QEC-1", "BEAD-QEC-2"],
      ["BEAD-QEC-1", "BEAD-QEC-3"],
      ...
    ]
  }
}
```

**Response**:
```json
{
  "rb_dominating_set": ["BEAD-QEC-4"],
  "dominating_set_size": 1,
  "critical_path": ["BEAD-QEC-4", "BEAD-QEC-1", "BEAD-QEC-2"],
  "blocked_tasks": ["BEAD-QEC-1", "BEAD-QEC-2", "BEAD-QEC-5", "BEAD-QEC-6"],
  "health_impact": 0.10
}
```

### 3. Treewidth Computation

**Endpoint**: `POST /api/validate/treewidth`

**Request**:
```json
{
  "graph": {
    "nodes": ["BEAD-QEC-1", "BEAD-QEC-2", ..., "BEAD-QEC-9"],
    "edges": [
      ["BEAD-QEC-1", "BEAD-QEC-2"],
      ["BEAD-QEC-1", "BEAD-QEC-3"],
      ...
    ]
  }
}
```

**Response**:
```json
{
  "treewidth": 2,
  "tree_decomposition": {
    "bags": [
      ["BEAD-QEC-1", "BEAD-QEC-2", "BEAD-QEC-3"],
      ["BEAD-QEC-2", "BEAD-QEC-4", "BEAD-QEC-5"]
    ],
    "width": 2
  },
  "coupling_level": "low",
  "health_impact": 0.08
}
```

### 4. FPT Parameter Validation

**Endpoint**: `POST /api/validate/fpt`

**Request**:
```json
{
  "parameter": "r",
  "value": 4,
  "problem": "r-IDS",
  "graph_size": 380
}
```

**Response**:
```json
{
  "parameter": "r",
  "value": 4,
  "complexity": "O(256n)",
  "concrete_bound": 97280,
  "fpt_class": "FPT",
  "optimal_for": "brain networks (LID ≈ 4-7)",
  "health_impact": 0.12
}
```

## MCP Tool Wrappers

### yada-services-secure MCP Tools

```python
# In yada-services-secure/src/mcp_server.py

@mcp_tool("validate_bipartite")
async def validate_bipartite_graph(
    requirements: List[str],
    tasks: List[str]
) -> Dict:
    """Validate bipartite graph coverage."""
    from merge2docs.src.backend.algorithms.bipartite_analysis import analyze_bipartite
    return analyze_bipartite(requirements, tasks)

@mcp_tool("validate_rb_domination")
async def validate_rb_domination(
    graph: Dict
) -> Dict:
    """Compute RB-dominating set for critical path."""
    from merge2docs.src.backend.algorithms.rb_domination import compute_rb_dominating_set
    return compute_rb_dominating_set(graph)

@mcp_tool("validate_treewidth")
async def validate_treewidth(
    graph: Dict
) -> Dict:
    """Compute treewidth for coupling analysis."""
    from merge2docs.src.backend.algorithms.treewidth import compute_treewidth
    return compute_treewidth(graph)

@mcp_tool("validate_fpt_complexity")
async def validate_fpt_complexity(
    parameter: str,
    value: int,
    problem: str,
    graph_size: int
) -> Dict:
    """Validate FPT complexity bounds."""
    from merge2docs.src.backend.algorithms.fpt_validator import validate_fpt
    return validate_fpt(parameter, value, problem, graph_size)
```

## Integration Script

**File**: `bin/validate_design_with_yada.py`

```python
#!/usr/bin/env python3
"""Validate QEC tensor design using yada-services-secure.

Usage:
    python bin/validate_design_with_yada.py \\
        --design docs/designs/yada-hierarchical-brain-model/DESIGN.md \\
        --requirements docs/designs/yada-hierarchical-brain-model/AUTO_REVIEW_QEC_TENSOR.md \\
        --output validation_report.json
"""

import argparse
import json
import httpx
from pathlib import Path
from typing import Dict, List

YADA_URL = "http://localhost:8003"


async def validate_design(
    requirements: List[str],
    tasks: List[str],
    graph: Dict
) -> Dict:
    """Run full design validation."""

    async with httpx.AsyncClient() as client:
        # 1. Bipartite graph analysis
        bipartite = await client.post(
            f"{YADA_URL}/api/validate/bipartite",
            json={"requirements": requirements, "tasks": tasks}
        )

        # 2. RB-domination
        rb_dom = await client.post(
            f"{YADA_URL}/api/validate/rb-domination",
            json={"graph": graph}
        )

        # 3. Treewidth
        treewidth = await client.post(
            f"{YADA_URL}/api/validate/treewidth",
            json={"graph": graph}
        )

        # 4. FPT validation
        fpt = await client.post(
            f"{YADA_URL}/api/validate/fpt",
            json={
                "parameter": "r",
                "value": 4,
                "problem": "r-IDS",
                "graph_size": 380
            }
        )

        # Compute health score
        health_score = (
            0.85 +  # Specification completeness
            bipartite.json()["health_impact"] +
            rb_dom.json()["health_impact"] +
            treewidth.json()["health_impact"] +
            fpt.json()["health_impact"]
        )

        return {
            "bipartite": bipartite.json(),
            "rb_domination": rb_dom.json(),
            "treewidth": treewidth.json(),
            "fpt": fpt.json(),
            "health_score": health_score,
            "timestamp": datetime.now().isoformat()
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Validate design using yada-services-secure"
    )
    parser.add_argument("--design", required=True, help="Design document path")
    parser.add_argument("--requirements", required=True, help="Requirements document path")
    parser.add_argument("--output", required=True, help="Output JSON path")

    args = parser.parse_args()

    # Parse design documents
    requirements = parse_requirements(Path(args.requirements))
    tasks = parse_tasks(Path(args.design))
    graph = build_dependency_graph(Path(args.design))

    # Validate
    results = await validate_design(requirements, tasks, graph)

    # Save report
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Validation complete: health score {results['health_score']:.2f}")
    print(f"📊 Report saved to {args.output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Implementation Steps

### Phase 1: Coordination (Week 2)

1. **Review merge2docs algorithms** (1 hour)
   - Check if algorithms are in working state
   - Verify they can be imported from yada-services-secure
   - Test locally with sample data

2. **Design HTTP endpoints** (2 hours)
   - API specification (see above)
   - Request/response schemas
   - Error handling

3. **Coordinate with merge2docs team** (async)
   - Share this specification
   - Confirm algorithms are stable
   - Discuss any API changes needed

### Phase 2: Implementation (Week 3)

4. **Implement HTTP endpoints in yada-services-secure** (4 hours)
   - Add routes to FastAPI app
   - Import merge2docs algorithms
   - Wrap with error handling
   - Add request validation

5. **Implement MCP tool wrappers** (2 hours)
   - Add MCP tool decorators
   - Test via MCP protocol
   - Document tool usage

6. **Create integration script** (2 hours)
   - `bin/validate_design_with_yada.py`
   - Parse design documents
   - Call validation endpoints
   - Generate report

### Phase 3: Testing (Week 3)

7. **Unit tests** (2 hours)
   - Test each endpoint independently
   - Mock merge2docs algorithms if needed
   - Verify error handling

8. **Integration tests** (2 hours)
   - Test with real QEC tensor design
   - Verify health score calculation
   - Test via MCP and HTTP

9. **Documentation** (1 hour)
   - API documentation
   - Usage examples
   - Update health score calculation

## Expected Results

### Health Score Impact

**Current Health Score**: 0.82 (82%)

**After Validation Tools**:
- Bipartite graph: +5% (automated requirements coverage)
- RB-domination: +5% (automated critical path)
- Treewidth: +8% (automated coupling analysis)
- FPT validation: +12% (automated complexity bounds)

**New Health Score**: 0.88 (88%) ✅ **TARGET REACHED!**

### Automation Benefits

1. **Instant validation** - No manual calculation
2. **Continuous monitoring** - Run on every design change
3. **Objective metrics** - Remove human interpretation
4. **Reproducible** - Same inputs = same outputs

## Deliverables

1. ✅ **API Specification** (this document)
2. ⏳ **HTTP Endpoints** in yada-services-secure
3. ⏳ **MCP Tool Wrappers** in yada-services-secure
4. ⏳ **Integration Script** (`bin/validate_design_with_yada.py`)
5. ⏳ **Documentation** and examples
6. ⏳ **Tests** (unit + integration)

## Timeline

- **Week 2**: Phase 1 (coordination) - 3 hours
- **Week 3**: Phase 2 (implementation) - 8 hours
- **Week 3**: Phase 3 (testing) - 5 hours

**Total**: 16 hours over 2 weeks

## Success Criteria

✅ All 4 validation tools exposed via HTTP and MCP
✅ Integration script runs successfully
✅ Health score calculated automatically
✅ Tests pass (unit + integration)
✅ Documentation complete

**Target**: Health score **0.88** (88%) ✅

## Dependencies

- merge2docs algorithms (already exist)
- yada-services-secure (already running on port 8003)
- FastAPI (already installed)
- MCP protocol support (already implemented)

## Risks

1. **Low Risk**: merge2docs algorithms might need updates
   - Mitigation: Review and test before integration

2. **Low Risk**: Import path conflicts
   - Mitigation: Use relative imports or PYTHONPATH

3. **Medium Risk**: Performance (complex graph algorithms)
   - Mitigation: Add caching for repeated validations

## Notes

This is a **LOW PRIORITY** task because:
- Design already validated manually (0.82 health score)
- Core implementation is complete and tested
- This adds automation, not functionality

However, it provides **HIGH VALUE** for:
- Continuous design validation
- Automated health score tracking
- Integration with CI/CD pipeline

**Recommendation**: Implement after core functionality is production-ready.

---

**Status**: ⏳ PENDING (coordination with merge2docs team)
**Priority**: LOW (Task 4 of 4)
**Expected Impact**: +6% health score (0.82 → 0.88)
