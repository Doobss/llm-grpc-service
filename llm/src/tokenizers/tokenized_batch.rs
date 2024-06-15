use crate::{BatchEncoding, GenerationBatch, GenerationRequest, Result, Tokenizer};
use candle_core::Tensor;
use indexmap::IndexMap;

#[derive(Debug)]
pub struct TokenizedBatch {
    pub requests: IndexMap<String, GenerationRequest>,
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub past_key_values: Option<Tensor>,
}

impl TokenizedBatch {
    pub fn from_generation_batch(batch: GenerationBatch, tokenizer: &Tokenizer) -> Result<Self> {
        let mut prompts: Vec<&str> = Vec::with_capacity(batch.len());
        let GenerationBatch { requests } = batch;
        for request in requests.values() {
            prompts.push(&request.content)
        }

        let BatchEncoding {
            ids,
            attention_mask,
        } = tokenizer.encode_batch(prompts, false)?;
        Ok(Self {
            requests,
            input_ids: ids,
            attention_mask,
            past_key_values: None,
        })
    }
}

impl TokenizedBatch {
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
