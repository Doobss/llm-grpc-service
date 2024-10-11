use crate::Result;
use candle_core::Tensor;

#[derive(Debug)]
pub struct BatchEncoding {
    pub ids: Tensor,
    pub token_ids: Vec<Vec<u32>>,
    pub attention_mask: Tensor,
}

impl BatchEncoding {
    pub fn token_length(&self) -> usize {
        return self.ids.shape().dims()[1];
    }

    pub fn append_tokens(&mut self, next_tokens: &Tensor) -> Result<()> {
        self.attention_mask = Tensor::cat(
            &[
                &self.attention_mask,
                &Tensor::ones(
                    next_tokens.shape().dims(),
                    self.attention_mask.dtype(),
                    self.attention_mask.device(),
                )?,
            ],
            1,
        )?;
        self.ids = Tensor::cat(&[&self.ids, next_tokens], 1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // use crate::{ModelType, Tokenizer};
}
