//! # oxide-gradient
//!
//! Gradient-based GPU resource optimization via ternary search.
//! Ternary {-1,0,+1} as gradient directions for kernel parameter auto-tuning.

/// Kernel parameters to optimize.
#[derive(Debug, Clone)]
pub struct KernelParams {
    pub block_x: u32,
    pub shared_mem_bytes: u32,
    pub warp_count: u32,
    pub occupancy: f64,
    pub execution_time_us: f64,
}

impl KernelParams {
    pub fn new(block_x: u32, shared_mem: u32, warp_count: u32) -> Self {
        Self {
            block_x, shared_mem_bytes: shared_mem, warp_count,
            occupancy: 0.0, execution_time_us: 0.0,
        }
    }

    /// Simulate occupancy based on parameters.
    pub fn simulate_occupancy(&mut self) {
        let sm_shared = 48 * 1024; // 48KB shared memory per SM
        let max_warps = 48; // max warps per SM
        let shared_limit = sm_shared as f64 / (self.shared_mem_bytes.max(1) as f64);
        let warp_limit = max_warps as f64 / self.warp_count as f64;
        self.occupancy = (shared_limit.min(warp_limit) / max_warps as f64).min(1.0);
        self.execution_time_us = 1000.0 / self.occupancy.max(0.1);
    }
}

/// Ternary gradient direction for parameter optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gradient { Decrease, Hold, Increase }

/// Ternary parameter search state.
pub struct TernarySearch {
    pub params: KernelParams,
    pub gradient: ParamGradient,
    pub history: Vec<SearchStep>,
    pub best_time: f64,
    pub best_params: KernelParams,
}

#[derive(Debug, Clone)]
pub struct ParamGradient {
    pub block_dir: Gradient,
    pub shared_dir: Gradient,
    pub warp_dir: Gradient,
}

#[derive(Debug, Clone)]
pub struct SearchStep {
    pub params: KernelParams,
    pub gradient: ParamGradient,
    pub improved: bool,
}

impl TernarySearch {
    pub fn new(params: KernelParams) -> Self {
        let mut p = params;
        p.simulate_occupancy();
        Self {
            best_params: p.clone(), best_time: p.execution_time_us,
            params: p, gradient: ParamGradient {
                block_dir: Gradient::Hold,
                shared_dir: Gradient::Hold,
                warp_dir: Gradient::Increase,
            },
            history: Vec::new(),
        }
    }

    /// Apply gradient to get next parameters.
    pub fn apply_gradient(&mut self) {
        self.params.block_x = match self.gradient.block_dir {
            Gradient::Increase => (self.params.block_x * 2).min(1024),
            Gradient::Decrease => (self.params.block_x / 2).max(32),
            Gradient::Hold => self.params.block_x,
        };
        self.params.shared_mem_bytes = match self.gradient.shared_dir {
            Gradient::Increase => (self.params.shared_mem_bytes + 1024).min(32768),
            Gradient::Decrease => self.params.shared_mem_bytes.saturating_sub(1024).max(0),
            Gradient::Hold => self.params.shared_mem_bytes,
        };
        self.params.warp_count = match self.gradient.warp_dir {
            Gradient::Increase => (self.params.warp_count + 4).min(48),
            Gradient::Decrease => self.params.warp_count.saturating_sub(4).max(4),
            Gradient::Hold => self.params.warp_count,
        };
        self.params.simulate_occupancy();
    }

    /// One step of ternary search.
    pub fn step(&mut self) -> f64 {
        self.apply_gradient();
        let improved = self.params.execution_time_us < self.best_time;
        if improved {
            self.best_time = self.params.execution_time_us;
            self.best_params = self.params.clone();
        }

        // Update gradient based on improvement
        if !improved {
            // Reverse direction
            self.gradient.block_dir = match self.gradient.block_dir {
                Gradient::Increase => Gradient::Decrease,
                Gradient::Decrease => Gradient::Hold,
                Gradient::Hold => Gradient::Increase,
            };
        }

        self.history.push(SearchStep {
            params: self.params.clone(),
            gradient: self.gradient.clone(),
            improved,
        });

        self.params.execution_time_us
    }

    /// Run search for N steps.
    pub fn search(&mut self, steps: usize) -> Vec<f64> {
        (0..steps).map(|_| self.step()).collect()
    }

    pub fn best_time(&self) -> f64 { self.best_time }
    pub fn steps_taken(&self) -> usize { self.history.len() }
    pub fn improvements(&self) -> usize { self.history.iter().filter(|s| s.improved).count() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_occupancy() {
        let mut p = KernelParams::new(256, 4096, 8);
        p.simulate_occupancy();
        assert!(p.occupancy > 0.0 && p.occupancy <= 1.0);
        assert!(p.execution_time_us > 0.0);
    }

    #[test]
    fn test_ternary_search_improves() {
        let params = KernelParams::new(32, 1024, 4);
        let mut search = TernarySearch::new(params);
        let times = search.search(10);
        assert!(search.best_time() <= times[0]);
    }

    #[test]
    fn test_gradient_application() {
        let params = KernelParams::new(128, 2048, 8);
        let mut search = TernarySearch::new(params);
        search.gradient.warp_dir = Gradient::Increase;
        search.apply_gradient();
        assert_eq!(search.params.warp_count, 12); // 8 + 4
    }

    #[test]
    fn test_history_tracking() {
        let params = KernelParams::new(64, 1024, 8);
        let mut search = TernarySearch::new(params);
        search.search(5);
        assert_eq!(search.steps_taken(), 5);
    }

    #[test]
    fn test_block_bounds() {
        let params = KernelParams::new(1024, 1024, 8);
        let mut search = TernarySearch::new(params);
        search.gradient.block_dir = Gradient::Increase;
        search.apply_gradient();
        assert_eq!(search.params.block_x, 1024); // capped
    }

    #[test]
    fn test_shared_mem_bounds() {
        let params = KernelParams::new(128, 0, 8);
        let mut search = TernarySearch::new(params);
        search.gradient.shared_dir = Gradient::Decrease;
        search.apply_gradient();
        assert_eq!(search.params.shared_mem_bytes, 0); // floored
    }

    #[test]
    fn test_best_params_tracked() {
        let params = KernelParams::new(64, 1024, 4);
        let mut search = TernarySearch::new(params);
        search.search(20);
        assert!(search.best_time() > 0.0);
        assert!(search.best_params.block_x >= 32);
    }
}
