# Parallel Bayesian Optimization

## Overview

The Bayesian optimization feature now supports **parallel execution** of parameter evaluations using multiple worker threads. This can significantly speed up optimization when you have sufficient computational resources.

## What Changed

### ✅ New Features

1. **`--max-workers` CLI Option**
   - Control the number of parallel worker threads
   - Default: `1` (sequential execution, original behavior)
   - Recommended: `2-4` workers for parallel execution

2. **Thread-Safe Result Recording**
   - All shared state updates (best parameters, iteration results) are protected by locks
   - Safe for concurrent execution

3. **Intelligent Ask/Tell Loop**
   - When `max_workers > 1`, uses scikit-optimize's `Optimizer.ask/tell` interface
   - Dynamically schedules evaluations across worker pool
   - Waits for batch completion before requesting next batch

### 🔧 Technical Implementation

#### Sequential Mode (`--max-workers 1`, default)
```python
# Uses gp_minimize (original behavior)
result = gp_minimize(
    objective,
    space,
    n_calls=n_iterations,
    random_state=42,
    n_random_starts=5
)
```

#### Parallel Mode (`--max-workers 2+`)
```python
# Uses Optimizer.ask/tell with ThreadPoolExecutor
opt = SkOptimizer(space, random_state=42, n_initial_points=5)
executor = ThreadPoolExecutor(max_workers=max_workers)

while len(iteration_results) < n_iterations:
    # Ask for batch of points
    points = [opt.ask() for _ in range(batch_size)]
    
    # Evaluate in parallel
    futures = [executor.submit(evaluate, x) for x in points]
    
    # Collect results and tell optimizer
    for future, x in zip(futures, points):
        y = future.result()
        opt.tell(x, y)
```

## Usage

### Basic Usage (Sequential)

```bash
# Default behavior - one evaluation at a time
opticonn bayesian \
  -i data/fib_samples \
  -o test/bayes \
  --config configs/user_friendly_sweep.json \
  --n-iterations 20
```

### Parallel Execution (2 Workers)

```bash
# Evaluate 2 parameter combinations simultaneously
opticonn bayesian \
  -i data/fib_samples \
  -o test/bayes_parallel \
  --config configs/user_friendly_sweep.json \
  --n-iterations 20 \
  --max-workers 2
```

### Aggressive Parallel (4 Workers)

```bash
# Evaluate 4 parameter combinations simultaneously (needs more resources)
opticonn bayesian \
  -i data/fib_samples \
  -o test/bayes_parallel \
  --config configs/user_friendly_sweep.json \
  --n-iterations 20 \
  --max-workers 4
```

## Performance Considerations

### When to Use Parallel Execution

✅ **Good Use Cases:**
- **You have multiple CPU cores available** (8+ cores recommended for `--max-workers 4`)
- **Each evaluation is I/O-bound** (waiting for tractography, file writes)
- **You want faster wall-clock time** (complete 20 iterations in less time)
- **Resource availability** (enough RAM for multiple DSI Studio instances)

❌ **When NOT to Use Parallel:**
- **Limited CPU cores** (< 4 cores → stick with sequential)
- **Memory constrained** (each worker needs ~4-8 GB RAM)
- **CPU-intensive evaluations** (will cause thrashing)
- **Debugging** (sequential is easier to trace)

### Expected Speedup

| Workers | Iterations | Sequential Time | Parallel Time | Speedup |
|---------|------------|-----------------|---------------|---------|
| 1       | 20         | ~10 hours       | ~10 hours     | 1.0x    |
| 2       | 20         | ~10 hours       | ~5.5 hours    | 1.8x    |
| 4       | 20         | ~10 hours       | ~3 hours      | 3.3x    |

*Note: Actual speedup depends on CPU, I/O, and evaluation overhead*

### Resource Requirements

#### Per Worker Resource Usage
- **CPU**: 1-2 cores per worker (DSI Studio is multi-threaded)
- **RAM**: 4-8 GB per worker
- **Disk I/O**: Moderate (writing iteration results, configs)

#### Recommended Configurations

**Conservative (2 workers):**
- CPU: 8+ cores
- RAM: 16+ GB
- Good for: Initial testing, moderate datasets

**Aggressive (4 workers):**
- CPU: 16+ cores
- RAM: 32+ GB
- Good for: Large-scale optimization, powerful workstations

## Thread Safety

All shared state modifications are protected by locks:

```python
with self._lock:
    self.iteration_results.append(result_record)
    
    if mean_qa > self.best_score:
        self.best_score = mean_qa
        self.best_params = params.copy()
    
    self._save_progress()
```

This ensures:
- No race conditions when recording results
- Consistent best parameter tracking
- Safe progress file updates

## Output Files

The output structure is identical for sequential and parallel modes:

```
test/bayes_parallel/
├── bayesian_optimization_results.json    # Final results
├── bayesian_optimization_progress.json   # Incremental progress
└── iterations/
    ├── iteration_0001_config.json
    ├── iteration_0001/
    │   ├── 01_connectivity/
    │   └── 02_optimization/
    ├── iteration_0002_config.json
    ├── iteration_0002/
    ...
```

**Note**: Iteration numbers may not complete in order when using parallel execution, but all results are tracked correctly.

## Example Command for Your Use Case

Based on your original request:

```bash
# Run 20 iterations with 2 parallel workers
opticonn bayesian \
  -i data/fib_samples \
  -o test/bayes \
  --config configs/user_friendly_sweep.json \
  --n-iterations 20 \
  --max-workers 2 \
  --verbose
```

This will:
- Evaluate 2 parameter combinations simultaneously
- Complete ~2x faster than sequential
- Show detailed progress with `--verbose`
- Save all results to `test/bayes/`

## Monitoring Parallel Execution

### Console Output

```
🚀 Starting Bayesian optimization...
⚡ Running with 2 parallel workers

======================================================================
🔬 Bayesian Iteration 1/20
======================================================================
Testing parameters:
  tract_count               = 161343
  fa_threshold              = 0.0958
  ...

======================================================================
🔬 Bayesian Iteration 2/20
======================================================================
Testing parameters:
  tract_count               = 73405
  fa_threshold              = 0.0857
  ...

✅ QA Score: 0.4175
🏆 New best QA score: 0.4175
💾 Progress saved to test/bayes/bayesian_optimization_progress.json
```

### Progress File

Monitor real-time progress:

```bash
watch -n 5 'cat test/bayes/bayesian_optimization_progress.json | jq ".completed_iterations, .best_score"'
```

## Troubleshooting

### Issue: "Out of memory" errors

**Solution**: Reduce `--max-workers` to 1 or 2

```bash
opticonn bayesian ... --max-workers 1
```

### Issue: Slower than sequential execution

**Possible causes**:
- Too many workers for available CPU cores
- CPU-bound evaluations (thrashing)
- Insufficient RAM (swapping)

**Solution**: Use fewer workers or stick with sequential execution

### Issue: Iteration numbers out of order in logs

**This is normal!** When running in parallel:
- Multiple iterations execute simultaneously
- Log messages may interleave
- Results are still recorded correctly
- Final results are accurate

## Benefits Summary

| Feature | Sequential (`--max-workers 1`) | Parallel (`--max-workers 2+`) |
|---------|-------------------------------|------------------------------|
| **Wall-clock time** | Baseline | 1.5-3.5x faster |
| **CPU usage** | Low | High |
| **Memory usage** | Low | High |
| **Debugging** | Easy | Harder |
| **Resource efficiency** | High | Moderate |
| **Best for** | Limited resources, debugging | Speed, powerful hardware |

## Implementation Files

- `scripts/bayesian_optimizer.py` - Core optimizer with parallel support
- `scripts/opticonn_hub.py` - CLI integration with `--max-workers` option

---

**Ready to use!** Start with `--max-workers 2` to test parallel execution on your system.
