use crate::{Model, ModelConfig, Result, TokenizedBatch, Tokenizer};
use candle_core::Tensor;

pub struct TextGeneration {
    pub model: Model,
    pub tokenizer: Tokenizer,
}

impl TextGeneration {
    pub fn next_token(&mut self, batch: &TokenizedBatch) -> Result<Tensor> {
        let logits = self.model.forward(batch)?.squeeze(1)?;
        Ok(logits)
    }
}

impl TextGeneration {
    pub fn new(config: ModelConfig) -> Result<Self> {
        Ok(Self {
            tokenizer: Tokenizer::load(config.model_id)?,
            model: Model::load(config)?,
        })
    }
}
