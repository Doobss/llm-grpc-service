use super::{Receiver, Sender, TaskResult};
use crate::{GenerationStep, TokenizedBatch, Tokenizer};

#[derive(Debug)]
pub struct Decoder {
    tokenizer: std::sync::Arc<Tokenizer>,
    generation_result_receiver: Receiver<GenerationStep>,
    tokenized_batch_sender: Sender<TokenizedBatch>,
}

impl Decoder {
    pub fn task(self) -> TaskResult<()> {
        let Decoder {
            tokenizer,
            tokenized_batch_sender,
            mut generation_result_receiver,
        } = self;
        tokio::task::spawn_blocking(move || loop {
            tracing::debug!("decode_task: awaiting results");
            if let Some(generation_result) = generation_result_receiver.blocking_recv() {
                let GenerationStep { batch, logits } = generation_result;
                let TokenizedBatch {
                    requests,
                    input_ids,
                    attention_mask,
                    past_key_values,
                } = batch;
                let mut indicies_to_keep = Vec::new();
                let end_of_sequence_tokens = crate::get_eos_tokens(&logits, tokenizer.eos_id)?;

                for (index, request) in requests.values().enumerate() {
                    let is_end_of_sequence = if let Some(value) = end_of_sequence_tokens.get(index)
                    {
                        *value == 1
                    } else {
                        false
                    };
                    if !is_end_of_sequence {
                        indicies_to_keep.push(index);
                    }
                }
            }
        })
    }
}

impl Decoder {
    pub fn new(
        tokenizer: std::sync::Arc<Tokenizer>,
        generation_result_receiver: Receiver<GenerationStep>,
        tokenized_batch_sender: Sender<TokenizedBatch>,
    ) -> Self {
        Self {
            tokenizer,
            generation_result_receiver,
            tokenized_batch_sender,
        }
    }
}
