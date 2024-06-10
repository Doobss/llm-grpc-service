use crate::{BatchEncoding, Model, ModelType, Result, Tokenizer};
use candle_core::Tensor;

pub struct TextGeneration {
    model: Model,
    pub tokenizer: Tokenizer,
}

impl TextGeneration {
    pub fn next(&mut self, batch: &BatchEncoding) -> Result<Tensor> {
        let logits = self
            .model
            .forward(&batch.ids, &batch.attention_mask)?
            .squeeze(1)?;
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
