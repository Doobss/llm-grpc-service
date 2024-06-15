use std::{any::Any, fmt::Debug};

use crate::Result;
use candle_core::Tensor;
use candle_transformers::generation::{self, Sampling};

#[derive()]
pub struct LogitsProcessor {
    inner: generation::LogitsProcessor,
}

impl LogitsProcessor {
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        Ok(self.inner.sample(logits)?)
    }
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        tracing::debug!(
            "Creating logits processor with {:?} sampling and a seed value {}",
            &sampling,
            &seed
        );
        let inner = generation::LogitsProcessor::from_sampling(seed, sampling);
        Self { inner }
    }

    pub fn from_config(config: &crate::PromptConfig) -> Self {
        let sampling = crate::utils::get_sampling(config.temperature, config.top_k, config.top_p);
        tracing::debug!(
            "Creating logits processor with {:?} sampling and a seed value {}",
            &sampling,
            &config.seed
        );

        let inner = generation::LogitsProcessor::from_sampling(config.seed, sampling);
        Self { inner }
    }
}

impl Default for LogitsProcessor {
    fn default() -> Self {
        let seed = 100;
        let sampling = generation::Sampling::ArgMax;
        let inner = generation::LogitsProcessor::from_sampling(seed, sampling);
        Self { inner }
    }
}

impl Debug for LogitsProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner_type = self.inner.type_id();
        f.debug_struct("LogitsProcessor")
            .field("inner", &inner_type)
            .finish()
    }
}

#[cfg(test)]
mod tests {}
