use crate::{PromptConfig, Result};
use candle_core::Tensor;
// use candle_transformers;

#[derive()]
pub struct LogitsPreProcessor {
    repetition_penalty: Option<f32>,
}

impl LogitsPreProcessor {
    pub fn from_config(config: &PromptConfig) -> Self {
        Self {
            repetition_penalty: config.repetition_penalty,
        }
    }

    pub fn process_logits(&self, logits: Tensor) -> Result<Tensor> {
        if let Some(_repetition_penalty) = self.repetition_penalty {
            // todo actually apply rep penalty
            Ok(logits)
        } else {
            Ok(logits)
        }
    }
}
