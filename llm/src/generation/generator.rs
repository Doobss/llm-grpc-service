extern crate tokio;
use super::{GenerationBatch, GenerationReply, GenerationRequest, TextGeneration};
use crate::{Error, ModelType, Prompt, Result};

pub type GenerationRequestSender = tokio::sync::mpsc::Sender<GenerationRequest>;
pub type GenerationReplyReceiver = tokio::sync::mpsc::Receiver<GenerationReply>;

pub struct Generator {
    request_sender: GenerationRequestSender,
}

type TaskResult = tokio::task::JoinHandle<Result<()>>;

impl Generator {
    pub async fn new(text_generation: TextGeneration) -> Self {
        let (request_sender, mut request_receiver) =
            tokio::sync::mpsc::channel::<GenerationRequest>(128);
        let (batch_sender, batch_receiver) = tokio::sync::mpsc::channel::<GenerationBatch>(128);

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

        let mut tokenize_task: TaskResult = tokio::task::spawn_blocking(move || loop {});

        let mut generation_task: TaskResult = tokio::task::spawn_blocking(move || loop {});

        let mut decode_task: TaskResult = tokio::task::spawn_blocking(move || loop {});

        tokio::select! {
            _ = (&mut batch_task) => {
                tracing::error!("Error in batch task, aborting all other tasks.");
                tokenize_task.abort();
                generation_task.abort();
                decode_task.abort();
            },
            _ = (&mut tokenize_task) => {
                tracing::error!("Error in tokenize task, aborting all other tasks.");
                batch_task.abort();
                generation_task.abort();
                decode_task.abort();
            },
            _ = (&mut generation_task) => {
                tracing::error!("Error in generation task, aborting all other tasks.");
                batch_task.abort();
                tokenize_task.abort();
                decode_task.abort();
            },
            _ = (&mut decode_task) => {
                tracing::error!("Error in decode task, aborting all other tasks.");
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
