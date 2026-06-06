# oxide-gradient

Gradient-based GPU resource optimization via ternary search.

## Why This Exists

GPU kernel performance is a function of three parameters: block size (threads per block), shared memory allocation, and warp count. The search space is small enough that you don't need full autotuning frameworks — you need a disciplined search that converges in <20 steps. But it's large enough that brute force (testing every combination) wastes GPU time on configurations that are clearly suboptimal.

The insight: treat optimization directions as ternary signals. Each parameter gets a direction: **Increase** (+1), **Hold** (0), or **Decrease** (-1). After each step, if performance improved, keep going. If not, reverse direction. This is ternary search with gradient-like momentum — it converges faster than grid search, requires no derivatives, and works on real hardware where profiling is expensive.

## Architecture

```
┌────────────────────────────────────────────┐
│            TernarySearch                    │
│                                            │
│  params: KernelParams                      │
│  ┌─────────────────────────────────────┐   │
│  │ block_x: 256                        │   │
│  │ shared_mem_bytes: 4096              │   │
│  │ warp_count: 8                       │   │
│  │ occupancy: 0.67  ← simulated       │   │
│  │ exec_time_us: 1492 ← simulated     │   │
│  └─────────────────────────────────────┘   │
│                                            │
│  gradient: ParamGradient                   │
│  ┌─────────────────────────────────────┐   │
│  │ block_dir:    Hold                  │   │
│  │ shared_dir:   Increase (+1)         │   │
│  │ warp_dir:     Increase (+1)         │   │
│  └─────────────────────────────────────┘   │
│                                            │
│  history: [SearchStep; N]                  │
│  best_time: 1100.0                         │
│  best_params: { block: 512, shared: 8192 } │
│                                            │
│  step() → f64 (new execution time)         │
│  search(n) → Vec<f64> (convergence curve)  │
└────────────────────────────────────────────┘

Occupancy Simulation:
  SM shared (48KB) / shared_mem → shared_limit
  Max warps (48) / warp_count  → warp_limit
  occupancy = min(shared_limit, warp_limit) / max_warps
  exec_time = 1000 / max(occupancy, 0.1)
```

**Key types:**

- `Gradient` — `Increase`, `Hold`, `Decrease` — the ternary direction signal
- `KernelParams` — block_x, shared_mem_bytes, warp_count + simulated occupancy and execution time
- `ParamGradient` — gradient direction for each parameter
- `SearchStep` — snapshot of params, gradient, and improvement flag at each step
- `TernarySearch` — the search engine

## Usage

```rust
use oxide_gradient::*;

// Start with conservative parameters
let initial = KernelParams::new(64, 1024, 4);
let mut search = TernarySearch::new(initial);

// Run 20 optimization steps
let convergence = search.search(20);

// Check results
println!("Best time: {:.0} μs", search.best_time());
println!("Best params: block={}, shared={}, warps={}",
    search.best_params.block_x,
    search.best_params.shared_mem_bytes,
    search.best_params.warp_count,
);
println!("Improvements found: {}/{}", search.improvements(), search.steps_taken());

// Manual step-by-step control
let params = KernelParams::new(256, 2048, 8);
let mut search = TernarySearch::new(params);
search.gradient.warp_dir = Gradient::Increase;

loop {
    let time = search.step();
    if search.steps_taken() >= 10 { break; }
}
```

## API Reference

### `KernelParams`

```rust
pub struct KernelParams {
    pub block_x: u32,           // Threads per block (32–1024, power of 2)
    pub shared_mem_bytes: u32,  // Shared memory per block (0–32768)
    pub warp_count: u32,        // Warps per SM (4–48)
    pub occupancy: f64,         // Simulated occupancy [0, 1]
    pub execution_time_us: f64, // Simulated execution time
}
```

- `new(block_x, shared_mem, warp_count) -> Self`
- `simulate_occupancy(&mut self)` — compute occupancy from SM shared memory limits and warp limits

### `Gradient`

```rust
pub enum Gradient { Decrease, Hold, Increase }
```

### `ParamGradient`

```rust
pub struct ParamGradient {
    pub block_dir: Gradient,
    pub shared_dir: Gradient,
    pub warp_dir: Gradient,
}
```

### `TernarySearch`

- `new(params: KernelParams) -> Self` — initialize with starting parameters
- `step() -> f64` — one optimization step, returns execution time
- `search(steps: usize) -> Vec<f64>` — run N steps, returns convergence curve
- `apply_gradient()` — apply current gradient to get next parameters
- `best_time() -> f64` / `steps_taken() -> usize` / `improvements() -> usize`

### `SearchStep`

```rust
pub struct SearchStep {
    pub params: KernelParams,
    pub gradient: ParamGradient,
    pub improved: bool,
}
```

## The Deeper Idea

This is the **optimization layer** in the oxide stack's performance architecture. The ternary gradient directions (Increase/Hold/Decrease) map to the same {-1, 0, +1} vocabulary used by health signals, capacity signals, and isolation quality throughout the ecosystem. A gradient that's been `Hold` for many steps is "balanced" — the same state as a balanced capacity node or a healthy GPU.

The occupancy model simulates the real constraint: shared memory per SM (48 KB) and max warps per SM (48). If each block uses 8 KB of shared memory, you can fit at most 6 blocks per SM. If each block has 8 warps, you can fit 6 blocks before hitting the warp limit. Occupancy is the min of these two ceilings, and execution time is inversely proportional. The optimizer searches for the sweet spot where both constraints are balanced.

## Related Crates

- **oxide-pipeline** — execution pipeline where optimized kernels run
- **oxide-compile-cache** — caches compiled kernels at their optimal parameters
- **oxide-energy-balance** — verifies that optimized kernels preserve algebraic invariants
- **oxide-checkpoint** — saves optimal parameter configurations for recovery
