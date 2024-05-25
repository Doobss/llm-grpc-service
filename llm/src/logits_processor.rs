use crate::Result;
use candle_core::Tensor;
use candle_transformers::generation;

#[derive()]
pub struct LogitsProcessor {
    inner: generation::LogitsProcessor,
}

impl LogitsProcessor {
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        Ok(self.inner.sample(logits)?)
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
