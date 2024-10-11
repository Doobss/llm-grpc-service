use super::{Receiver, Sender, TaskResult};
use crate::{GenerationBatch, TokenizedBatch, Tokenizer};
use std::sync::Arc;

#[derive(Debug)]
pub struct Tokenize {
    tokenizer: std::sync::Arc<Tokenizer>,
    generation_batch_receiver: Receiver<GenerationBatch>,
    tokenized_batch_sender: Sender<TokenizedBatch>,
}

impl Tokenize {
    pub fn task(self) -> TaskResult<()> {
        let Tokenize {
            tokenizer,
            mut generation_batch_receiver,
            tokenized_batch_sender,
        } = self;
        tokio::task::spawn_blocking(move || {
            let tokenizer = tokenizer.as_ref();
            loop {
                tracing::debug!("tokenize_task: awaiting batches");
                if let Some(generation_batch) = generation_batch_receiver.blocking_recv() {
                    let loop_start = tokio::time::Instant::now();
                    let tokenized_batch =
                        TokenizedBatch::from_generation_batch(generation_batch, tokenizer)?;
                    tracing::debug!("tokenize_task: sending batch {:?}", &tokenized_batch);
                    tokenized_batch_sender.blocking_send(tokenized_batch)?;
                    let loop_end = loop_start.elapsed().as_micros();
                    tracing::debug!("tokenize task finished in: {:?} micro seconds", loop_end);
                }
            }
        })
    }
}

impl Tokenize {
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        generation_batch_receiver: Receiver<GenerationBatch>,
        tokenized_batch_sender: Sender<TokenizedBatch>,
    ) -> Self {
        Self {
            tokenizer,
            generation_batch_receiver,
            tokenized_batch_sender,
        }
    }
}
