# Design: Service Layer (F₃)

**Task ID**: PH-3  
**Functor Level**: F₃ (Service)
**Biological Level**: Level 3-4 (Regulatory → Epigenetic)
**Priority**: High
**Status**: ✅ Complete

## Overview

F₃ level provides composed services for LOC simulation, sensing, and MRI analysis orchestration.

## Deliverables

### Completed (2026-01-20)
- ✅ `src/backend/services/_service_base.py` - Base classes, exceptions
- ✅ `src/backend/services/loc_simulator.py` - LOCSimulator service
- ✅ `src/backend/services/sensing_service.py` - SensingService
- ✅ `src/backend/services/mri_analysis_orchestrator.py` - MRI orchestrator
- ✅ `src/backend/services/config_schema.yaml` - Configuration schema
- ✅ `tests/test_services.py` - 15 unit tests (100% pass)

## Features

- Async service architecture with FPT bounds
- Parameter validation (wavelength, NA, pixel size, dark signal)
- Health check contract for all services
- Three-tier exception hierarchy (400, 500, 503)

## References

- Implementation: Commit 5488ee7
- Design docs: `../design-ph.1-physics_architecture/implementation_guidance.md`
