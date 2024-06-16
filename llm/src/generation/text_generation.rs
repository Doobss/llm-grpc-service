use crate::{Model, ModelType, Result, TokenizedBatch, Tokenizer};
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
    pub fn new(model_type: ModelType) -> Result<Self> {
        Ok(Self {
            model: Model::load(model_type)?,
            tokenizer: Tokenizer::load(model_type)?,
        })
    }
}
