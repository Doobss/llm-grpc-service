use crate::Result;
use candle_core::Tensor;
use candle_examples::device as get_device;
use tokenizers::{Encoding, PaddingParams};

#[derive(Debug)]
pub struct BatchEncoding {
    pub keys: Vec<String>,
    pub ids: Tensor,
    pub attention_mask: Tensor,
    padding: PaddingParams,
}

impl BatchEncoding {
    pub fn from_encodings(
        keys: Vec<String>,
        encodings: Vec<Encoding>,
        padding: &PaddingParams,
    ) -> Result<Self> {
        let mut ids: Vec<&[u32]> = Vec::new();
        let mut attentions: Vec<&[u32]> = Vec::new();
        for encoding in encodings.iter() {
            ids.push(encoding.get_ids());
            attentions.push(encoding.get_attention_mask());
        }
        let device = get_device(false)?;
        Ok(Self {
            keys,
            ids: Tensor::new(ids, &device)?,
            attention_mask: Tensor::new(attentions, &device)?,
            padding: padding.clone(),
        })
    }

    pub fn token_length(&self) -> usize {
        return self.ids.shape().dims()[1];
    }

    pub fn append_tokens(&mut self, next_tokens: &Tensor) -> Result<()> {
        self.attention_mask = Tensor::cat(
            &[
                &self.attention_mask,
                &Tensor::ones(
                    next_tokens.shape().dims(),
                    next_tokens.dtype(),
                    next_tokens.device(),
                )?,
            ],
            1,
        )?;
        self.ids = Tensor::cat(&[&self.ids, next_tokens], 1)?;
        Ok(())
    }

    pub fn merge_batch(&mut self, mut other_batch: BatchEncoding) -> Result<()> {
        match (self.token_length(), other_batch.token_length()) {
            (self_length, other_length) if self_length > other_length => {
                let added_tokens = self_length.saturating_sub(other_length);
                let shape = other_batch.ids.shape().clone();
                let dims = shape.dims();
                tracing::debug!("dims {:?}", dims);
                tracing::debug!("added_tokens {:?}", added_tokens);
                let added_tensor = Tensor::full(
                    other_batch.padding.pad_id,
                    (dims[0], added_tokens),
                    other_batch.ids.device(),
                )?;
                other_batch.ids = Tensor::cat(&[&added_tensor, other_batch.get_ids_ref()], 1)?;
                let added_tensor = Tensor::ones(
                    (dims[0], added_tokens),
                    other_batch.get_attention_mask_ref().dtype(),
                    other_batch.ids.device(),
                )?;
                other_batch.attention_mask =
                    Tensor::cat(&[&added_tensor, other_batch.get_attention_mask_ref()], 1)?;
                tracing::debug!("END other_batch {:?}", &other_batch);
            }
            (self_length, other_length) if other_length > self_length => {
                let added_tokens = other_length.saturating_sub(self_length);
                let shape = self.ids.shape().clone();
                let dims = shape.dims();
                tracing::debug!("dims {:?}", dims);
                tracing::debug!("added_tokens {:?}", added_tokens);
                let added_tensor = Tensor::full(
                    self.padding.pad_id,
                    (dims[0], added_tokens),
                    self.ids.device(),
                )?;

                self.ids = Tensor::cat(&[&added_tensor, self.get_ids_ref()], 1)?;
                let added_tensor = Tensor::zeros(
                    (dims[0], added_tokens),
                    self.get_attention_mask_ref().dtype(),
                    self.ids.device(),
                )?;
                self.attention_mask =
                    Tensor::cat(&[&added_tensor, self.get_attention_mask_ref()], 1)?;
                tracing::debug!("END self {:?}", &self);
            }
            _ => (),
        }
        tracing::debug!("CAT self {:?} | other {:?}", &self, &other_batch);
        self.ids = Tensor::cat(&[&self.ids, &other_batch.ids], 0)?;
        self.attention_mask = Tensor::cat(&[&self.attention_mask, &other_batch.attention_mask], 0)?;
        self.keys.extend(other_batch.keys);
        Ok(())
    }
}

impl BatchEncoding {
    pub fn set_ids(&mut self, new_ids: Tensor) {
        self.ids = new_ids;
    }

    pub fn get_ids_ref(&self) -> &Tensor {
        &self.ids
    }

    pub fn get_mut_ids_ref(&mut self) -> &Tensor {
        &self.ids
    }

    pub fn set_attention_mask(&mut self, new_attention_mask: Tensor) {
        self.attention_mask = new_attention_mask
    }

    pub fn get_attention_mask_ref(&self) -> &Tensor {
        &self.attention_mask
    }

    pub fn get_mut_attention_mask_ref(&mut self) -> &Tensor {
        &self.attention_mask
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ModelType, Prompt, Tokenizer};

    #[test]
    fn batch_merge_same_size() {
        let model_type = ModelType::Mistral7bInstructV02;
        let tokenizer = Tokenizer::load(model_type).expect("Error loading tokenizer");
        let mut batch = tokenizer
            .encode_batch(vec![Prompt {
                id: "1".to_owned(),
                content: "last test I swear".to_owned(),
            }])
            .expect("Error encoding batch");
        let input_batch = tokenizer
            .encode_batch(vec![
                Prompt {
                    id: "1".to_owned(),
                    content: "last test I swear".to_owned(),
                },
                Prompt {
                    id: "2".to_owned(),
                    content: "last test I swear".to_owned(),
                },
            ])
            .expect("Error encoding batch");
        batch.merge_batch(input_batch).expect("");
    }

    #[test]
    fn batch_merge_longer() {
        let model_type = ModelType::Mistral7bInstructV02;
        let tokenizer = Tokenizer::load(model_type).expect("Error loading tokenizer");
        let mut batch = tokenizer
            .encode_batch(vec![
                Prompt {
                    id: "1".to_owned(),
                    content: "last test I swear".to_owned(),
                },
                Prompt {
                    id: "2".to_owned(),
                    content: "last test I swear".to_owned(),
                },
            ])
            .expect("Error encoding batch");
        let input_batch = tokenizer.encode_batch(vec![
            Prompt {
                id: "3".to_owned(),
                content: "test test test test test test test another another another another another another anotheranother another and another and another".to_owned(),
            }]).expect("Error encoding batch");
        batch.merge_batch(input_batch).expect("");
    }

    #[test]
    fn batch_merge_smaller() {
        let model_type = ModelType::Mistral7bInstructV02;
        let tokenizer = Tokenizer::load(model_type).expect("Error loading tokenizer");
        let mut batch = tokenizer.encode_batch(vec![Prompt {
            id: "3".to_owned(),
            content: "test test test test test test test another another another another another another anotheranother another and another and another".to_owned(),
        }]).expect("Error encoding batch");
        let input_batch = tokenizer
            .encode_batch(vec![
                Prompt {
                    id: "1".to_owned(),
                    content: "last test I swear".to_owned(),
                },
                Prompt {
                    id: "2".to_owned(),
                    content: "last test I swear".to_owned(),
                },
            ])
            .expect("Error encoding batch");
        batch.merge_batch(input_batch).expect("");
    }
}
