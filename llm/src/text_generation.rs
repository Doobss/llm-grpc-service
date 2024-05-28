use crate::{BatchEncoding, LogitsProcessor, Model, ModelType, Result, Tokenizer};
use candle_core::Tensor;

pub struct TextGeneration {
    model: Model,
    pub tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl TextGeneration {
    pub fn next(&mut self, batch: &BatchEncoding) -> Result<Tensor> {
        let logits = self
            .model
            .forward(&batch.ids, &batch.attention_mask)?
            .squeeze(1)?;
        // tracing::info!("logits: {:?}", &logits);
        let next_tokens = self.logits_processor.sample(&logits)?;
        Ok(next_tokens)
    }
}

impl TextGeneration {
    pub fn new(model_type: ModelType, sampling: Option<crate::Sampling>) -> Result<Self> {
        let logits_processor = match sampling {
            Some(sampling) => LogitsProcessor::from_sampling(100, sampling),
            None => LogitsProcessor::default(),
        };
        Ok(Self {
            logits_processor,
            model: Model::load(model_type)?,
            tokenizer: Tokenizer::load(model_type)?,
        })
    }
}
