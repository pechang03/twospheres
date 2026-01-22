# Memory Audit and Fixes

**Date**: 2026-01-22
**Status**: Critical Memory Leak Fixes

---

## Issues Found and Fixed

### ✅ FIXED: P3 Path Materialization (twosphere-mcp commit 6e200c7)
**Location**: `src/backend/mri/tripartite_multiplex_analysis.py`
**Problem**: Materialized ALL P3 paths: O(|A| × |B| × |C|) tuples
**Fix**: Lookup dict approach O(|A| × |C|)
**Impact**: Millions of paths → thousands of reachability pairs

### ✅ FIXED: Euler Training Pairs (merge2docs commit b6de32b6)
**Location**: `src/backend/algorithms/gnn_euler_embedding.py`
**Problem**: Full one-hot vectors: 58MB for N=368, 10K pairs
**Fix**: Store indices instead: `(vertex_idx, neighbor_indices)`
**Impact**: 58MB → 160KB (360× reduction)

### ✅ FIXED: FastMap Memory Leak (twosphere-mcp commit f7cfd08)
**Location**: `src/backend/mri/fast_obstruction_detection.py`
**Problem**: FastMap graphs created but never deleted
**Fix**: Added try/finally with `bridge.delete_fastmap_graph(graph_id)`
**Impact**: Prevents accumulation of ~47KB+ per detection

---

## Issues Found - NEEDS FIXING

### ⚠️ CRITICAL: FFT Operations Without Proper Cleanup

#### 1. Quantum Fourier Features (merge2docs)
**Location**: `src/backend/algorithms/quantum_fourier_features.py:192`
```python
fft_coeffs = np.fft.fft(token_sequence)  # Creates complex array
frequencies = np.fft.fftfreq(n)          # Creates frequency array
```
**Issue**: Called repeatedly, arrays not explicitly deleted
**Fix Needed**:
- Limit array sizes
- Use generator pattern
- Add explicit cleanup after use
**Severity**: MEDIUM (numpy arrays, but repeated calls accumulate)

#### 2. Orch-OR FFT Projections (merge2docs)
**Location**: `src/backend/gnn/orchor_gnn.py:81,99`
```python
x_freq = torch.fft.rfft(x, dim=-1)     # Forward FFT
x_reconstructed = torch.fft.irfft(...)  # Inverse FFT
```
**Issue**: PyTorch gradient graph may accumulate if not in `torch.no_grad()` mode
**Audit Needed**:
- ✅ `lean_qtrm_service.py:269,319` - Uses `torch.no_grad()` (SAFE)
- ✅ `artist_qec_service.py:248` - Uses `model.eval()` (SAFE)
- ❓ **Need to audit ALL callers** to ensure `torch.no_grad()` context

**Fix Needed**: Add assertions in forward() to detect gradient tracking during inference

---

## Recommended Fixes

### 1. Add Inference Mode Assertions (Orch-OR GNN)
```python
# In FourierSpaceProjection.forward()
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        # Safe inference mode
        pass
    else:
        # Warn if gradients enabled during inference
        logger.warning("FFT projection called with gradients enabled - memory may accumulate")

    # ... existing code ...
```

### 2. Explicit Cleanup for Quantum Fourier Features
```python
def extract_token_fft_features(self, tokens: List[str]) -> Dict[str, Any]:
    try:
        # ... FFT computation ...
        fft_coeffs = np.fft.fft(token_sequence)

        # Store only what's needed, discard rest
        result = {
            'token_fft_coefficients': fft_coeffs[:self.num_modes].copy(),
            # ...
        }

        # Explicitly delete large arrays
        del fft_coeffs
        del token_sequence

        return result
    finally:
        # Ensure cleanup
        import gc
        gc.collect()
```

### 3. Memory Monitoring Decorator
```python
def monitor_memory(func):
    """Decorator to log memory usage before/after function calls."""
    import psutil
    import os

    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        delta = mem_after - mem_before

        if delta > 10:  # Log if >10MB increase
            logger.warning(f"{func.__name__} increased memory by {delta:.1f}MB")

        return result
    return wrapper

# Apply to suspect functions:
@monitor_memory
def extract_token_fft_features(self, tokens):
    ...
```

---

## Audit Checklist

### Models in Running Code
- [x] Q-mamba (QEC-ComoRAG) - **LOW RISK**: Small loops (max 3 cycles), lightweight objects
- [x] Orch-OR GNN FFT - **MEDIUM RISK**: Check all callers use `torch.no_grad()`
- [ ] **TODO**: Audit QTRM router service
- [ ] **TODO**: Audit GPU graph manager
- [ ] **TODO**: Audit artist agent services

### Common Patterns to Check
- [ ] Graph creation in loops without cleanup
- [ ] `.copy()` operations in tight loops
- [ ] `.subgraph()` operations (creates new graph objects)
- [ ] `.todense()` / `.toarray()` on sparse matrices
- [ ] FastMap/embedding creation without deletion
- [ ] Accumulating lists/dicts without bounds

---

## Testing Strategy

### 1. Memory Profiling Test
```python
import tracemalloc
import psutil
import os

def profile_memory_leak(func, iterations=100):
    """Profile function for memory leaks."""
    tracemalloc.start()
    process = psutil.Process(os.getpid())

    baseline = process.memory_info().rss / 1024 / 1024

    for i in range(iterations):
        func()

        if i % 10 == 0:
            current = process.memory_info().rss / 1024 / 1024
            delta = current - baseline
            print(f"Iteration {i}: {delta:.1f}MB increase")

            if delta > 500:  # >500MB leak
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("\nTop 10 memory consumers:")
                for stat in top_stats[:10]:
                    print(stat)
                break

    tracemalloc.stop()
```

### 2. Integration Test with Monitoring
```bash
# Run merge2docs with memory monitoring
python -m memory_profiler src/backend/services/lean_qtrm_service.py

# Watch for growth over time
watch -n 1 'ps aux | grep python | grep -v grep'
```

---

## Priority Actions

1. **IMMEDIATE**: Audit all Orch-OR GNN callers for `torch.no_grad()` context
2. **HIGH**: Add memory monitoring to quantum fourier feature extraction
3. **MEDIUM**: Add cleanup to FFT operations with explicit `del` and `gc.collect()`
4. **LOW**: Add memory profiling tests to CI/CD

---

## Commits

- `6e200c7` - Fix P3 path materialization (twosphere-mcp)
- `b6de32b6` - Fix Euler training pairs (merge2docs)
- `f7cfd08` - Fix FastMap memory leak (twosphere-mcp)
- **TODO**: Fix quantum fourier cleanup
- **TODO**: Add inference mode assertions

