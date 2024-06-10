extern crate tokio;
use std::sync::Arc;
use super::{GenerationBatch, GenerationReply, GenerationRequest, TextGeneration};
use crate::{logits, Error, ModelType, Prompt, Result, TokenizedBatch};
use candle_core::Tensor;

pub type GenerationRequestSender = tokio::sync::mpsc::Sender<GenerationRequest>;
pub type GenerationReplyReceiver = tokio::sync::mpsc::Receiver<GenerationReply>;

pub struct Generator {
    request_sender: GenerationRequestSender,
}

type TaskResult = tokio::task::JoinHandle<Result<()>>;

impl Generator {
    pub async fn new(text_generation: TextGeneration) -> Self {
        use tokio::sync::mpsc::channel;
        let tokenizer = Arc::new(text_generation.tokenizer);

        let (request_sender, mut request_receiver) = channel::<GenerationRequest>(128);
        let (batch_sender, mut batch_receiver) = channel::<GenerationBatch>(128);
        let (generation_sender, mut generation_receiver) = channel::<TokenizedBatch>(128);
        let (decode_sender, mut decode_receiver) = channel::<(TokenizedBatch, Tensor)>(128);

        let mut batch_task: TaskResult = tokio::spawn(async move {
            loop {
                let mut requests = Vec::new();
                tracing::debug!("batch_task: awaiting requests");
                match request_receiver.recv().await {
                    Some(request) => requests.push(request),
                    None => {
                        return Err(Error::GenerationError {
                            message: "Request receiver is closed. Restarting generator".to_owned(),
                        })
                    }
                };
                while !request_receiver.is_empty() {
                    match request_receiver.recv().await {
                        Some(request) => requests.push(request),
                        None => {
                            return Err(Error::GenerationError {
                                message: "Request receiver is closed. Restarting generator"
                                    .to_owned(),
                            })
                        }
                    }
                }
                let new_batch = GenerationBatch::from_requests(requests);
                tracing::debug!("batch_task: adding new batch of len: {}", &new_batch.len());
                batch_sender.send(new_batch).await?;
            }
        });

        let tokenizer_binding = tokenizer.clone();
        let tokenizer_generation_sender = generation_sender.clone();
        let mut tokenize_task: TaskResult = tokio::task::spawn_blocking(move || {
            let tokenizer = tokenizer_binding.as_ref();
            loop {
                tracing::debug!("tokenize_task: awaiting batches");
                if let Some(generation_batch) = batch_receiver.blocking_recv() {
                    let tokenized_batch = TokenizedBatch::from_generation_batch(generation_batch, tokenizer)?;
                    tracing::debug!("tokenize_task: sending batch {:?}", &tokenized_batch);
                    tokenizer_generation_sender.blocking_send(tokenized_batch)?;
                }
            }
        });

        let mut model = text_generation.model;
        let mut generation_task: TaskResult = tokio::task::spawn_blocking(move || {
            loop {
                tracing::debug!("generation_task: awaiting batches");
                if let Some(tokenized_batch) = generation_receiver.blocking_recv() {
                    let next_tokens = model.forward(&tokenized_batch)?;
                    decode_sender.blocking_send((tokenized_batch, next_tokens))?;
                }
            }
        });

        let decode_binding = tokenizer.clone();
        let decode_generation_sender = generation_sender.clone();
        let mut decode_task: TaskResult = tokio::task::spawn_blocking(move || {
            let tokenizer = decode_binding.as_ref();
            loop {
                tracing::debug!("decode_task: awaiting results");
                if let Some((tokenized_batch, logits)) = decode_receiver.blocking_recv() {
                    let TokenizedBatch {
                        requests,
                        input_ids,
                        attention_mask,
                        past_key_values
                    } = tokenized_batch;
                    let mut indicies_to_keep = Vec::new();
                    let end_of_sequence_tokens = crate::get_eos_tokens(&next_tokens, tokenizer.eos_id)?;

                    for (index, request) in requests.values().enumerate() {
                        let is_end_of_sequence = if let Some(value) = end_of_sequence_tokens.get(index) {
                            *value == 1
                        } else {
                            false
                        };
                        if !is_end_of_sequence {
                            indicies_to_keep.push(index);
                        }


                    }
                }
            }
        });

        tokio::select! {
            _ = (&mut batch_task) => {
                if let Err(error) = batch_task.await {
                    tracing::error!("Error in batch task, aborting all other tasks. {:?}", error);
                }
                tokenize_task.abort();
                generation_task.abort();
                decode_task.abort();
            },
            _ = (&mut tokenize_task) => {
                if let Err(error) = tokenize_task.await {
                    tracing::error!("Error in tokenize task, aborting all other tasks. {:?}", error);
                }
                batch_task.abort();
                generation_task.abort();
                decode_task.abort();
            },
            _ = (&mut generation_task) => {
                if let Err(error) = generation_task.await {
                    tracing::error!("Error in generation task, aborting all other tasks. {:?}", error);
                }
                batch_task.abort();
                tokenize_task.abort();
                decode_task.abort();
            },
            _ = (&mut decode_task) => {
                if let Err(error) = decode_task.await {
                    tracing::error!("Error in decode task, aborting all other tasks. {:?}", error);
                }
                batch_task.abort();
                tokenize_task.abort();
                generation_task.abort();
            },
        }
        Self { request_sender }
    }

    pub async fn from_model_type(model_type: ModelType) -> Result<Self> {
        let generation = TextGeneration::new(model_type)?;
        Ok(Generator::new(generation).await)
    }
}

impl Generator {
    pub async fn prompt(&self, prompt: Prompt) -> Result<GenerationReplyReceiver> {
        let (reply_sender, reply_receiver) = tokio::sync::mpsc::channel::<GenerationReply>(128);
        let generation_request = GenerationRequest::from_prompt(prompt, reply_sender);
        self.request_sender.send(generation_request).await?;
        Ok(reply_receiver)
    }
}
