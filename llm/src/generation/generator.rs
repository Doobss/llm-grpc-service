extern crate tokio;
use candle_core::cuda::cudarc::cublaslt::result;

use super::{GenerationBatch, GenerationRequest, GenerationResult, TextGeneration};
use crate::{tasks, GenerationStep, ModelType, Prompt, Result, TokenizedBatch};
use std::sync::Arc;

pub type GenerationRequestSender = tokio::sync::mpsc::Sender<GenerationRequest>;
pub type GenerationResultReceiver = tokio::sync::mpsc::Receiver<GenerationResult>;

#[derive(Debug)]
pub struct Generator {
    request_sender: GenerationRequestSender,
}

impl Generator {
    pub async fn new(text_generation: TextGeneration) -> Self {
        use tokio::sync::mpsc::channel;
        let TextGeneration { model, tokenizer } = text_generation;
        let tokenizer = Arc::new(tokenizer);

        let (request_sender, request_receiver) = channel::<GenerationRequest>(128);
        let (generation_batch_sender, generation_batch_receiver) = channel::<GenerationBatch>(128);
        let (tokenized_batch_sender, tokenized_batch_receiver) = channel::<TokenizedBatch>(128);
        let (generation_result_sender, generation_result_receiver) = channel::<GenerationStep>(128);

        let batch_task = tasks::Batching::new(request_receiver, generation_batch_sender);
        let tokenize_task = tasks::Tokenize::new(
            tokenizer.clone(),
            generation_batch_receiver,
            tokenized_batch_sender.clone(),
        );
        let generation_task =
            tasks::Generation::new(model, tokenized_batch_receiver, generation_result_sender);

        let decode_task = tasks::Decoder::new(
            tokenizer.clone(),
            generation_result_receiver,
            tokenized_batch_sender,
        );

        let mut batch_task = batch_task.task();
        let mut tokenize_task = tokenize_task.task();
        let mut generation_task = generation_task.task();
        let mut decode_task = decode_task.task();

        tokio::spawn(async move {
            tokio::select! {
                result = (&mut batch_task) => {
                    if let Err(error) = result {
                        tracing::error!("Error in batch task, aborting all other tasks. {:?}", error);
                    }
                    // tokenize_task.abort();
                    // generation_task.abort();
                    // decode_task.abort();
                },
                result = (&mut tokenize_task) => {
                    if let Err(error) = result {
                        tracing::error!("Error in tokenize task, aborting all other tasks. {:?}", error);
                    }
                    // batch_task.abort();
                    // generation_task.abort();
                    // decode_task.abort();
                },
                result = (&mut generation_task) => {
                    if let Err(error) = result {
                        tracing::error!("Error in generation task, aborting all other tasks. {:?}", error);
                    }
                    // batch_task.abort();
                    // tokenize_task.abort();
                    // decode_task.abort();
                },
                result = (&mut decode_task) => {
                    if let Err(error) = result {
                        tracing::error!("Error in decode task, aborting all other tasks. {:?}", error);
                    }
                    // batch_task.abort();
                    // tokenize_task.abort();
                    // generation_task.abort();
                },
            }
        });

        tracing::debug!("All generator tasks setup.");
        Self { request_sender }
    }

    pub async fn from_model_type(model_type: ModelType) -> Result<Self> {
        let generation = TextGeneration::new(model_type)?;
        Ok(Generator::new(generation).await)
    }
}

impl Generator {
    pub async fn prompt(&self, prompt: Prompt) -> Result<GenerationResultReceiver> {
        let (reply_sender, reply_receiver) = tokio::sync::mpsc::channel::<GenerationResult>(128);
        let generation_request = GenerationRequest::from_prompt(prompt, reply_sender);
        self.request_sender.send(generation_request).await?;
        Ok(reply_receiver)
    }
}
