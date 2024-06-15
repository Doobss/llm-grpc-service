use super::{Receiver, Sender, TaskResult};
use crate::{GenerationStep, Model, TokenizedBatch};

#[derive(Debug)]
pub struct Generation {
    model: Model,
    tokenized_batch_receiver: Receiver<TokenizedBatch>,
    generation_result_sender: Sender<GenerationStep>,
}

impl Generation {
    pub fn task(self) -> TaskResult<()> {
        let Generation {
            mut model,
            mut tokenized_batch_receiver,
            generation_result_sender,
        } = self;
        tokio::task::spawn_blocking(move || loop {
            tracing::debug!("generation_task: awaiting batches");
            if let Some(batch) = tokenized_batch_receiver.blocking_recv() {
                let loop_start = tokio::time::Instant::now();
                let logits = model.forward(&batch)?;
                generation_result_sender.blocking_send(GenerationStep { batch, logits })?;
                let loop_end = loop_start.elapsed().as_micros();
                tracing::debug!("generation task finished in: {:?} micro seconds", loop_end);
            }
        })
    }
}

impl Generation {
    pub fn new(
        model: Model,
        tokenized_batch_receiver: Receiver<TokenizedBatch>,
        generation_result_sender: Sender<GenerationStep>,
    ) -> Self {
        Self {
            model,
            tokenized_batch_receiver,
            generation_result_sender,
        }
    }
}
