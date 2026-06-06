# oxide-gradient

Experiment: gradient-based GPU resource optimization via ternary search. Ternary {-1,0,+1} as gradient directions for block size, shared memory, and warp count auto-tuning.

## Overview

# oxide-gradient

Gradient-based GPU resource optimization via ternary search.

## Stats

- **Tests**: 7
- **LOC**: 198
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use oxide_gradient::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
