use crate::Result;
use candle_core::Tensor;
use candle_transformers::generation;

#[derive()]
pub struct LogitsProcessor {
    sampling: generation::Sampling,
    inner: generation::LogitsProcessor,
}

impl LogitsProcessor {
    pub fn sample(&mut self, logits: &Tensor) -> Result<Tensor> {
        match self.sampling {
            generation::Sampling::ArgMax => Ok(logits.argmax_keepdim(1)?),
            _ => Ok(Tensor::new(
                vec![self.inner.sample(logits)?],
                logits.device(),
            )?),
        }
    }
}

impl Default for LogitsProcessor {
    fn default() -> Self {
        let seed = 100;
        let sampling = generation::Sampling::ArgMax;
        let inner = generation::LogitsProcessor::from_sampling(seed, sampling.clone());
        Self { inner, sampling }
    }
}
