use crate::services::v1::*;
use std::{collections::HashMap, pin::Pin};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{Request, Response, Status};

type LlmResult<T> = Result<Response<T>, Status>;
type ResponseStream = Pin<Box<dyn Stream<Item = Result<PromptReply, Status>> + Send>>;

#[derive(Debug)]
pub struct LlmServer {
    generator: TextGenerator,
}

impl LlmServer {
    pub fn new(model_type: ModelType) -> crate::Result<Self> {
        Ok(Self {
            generator: TextGenerator::new(model_type).expect("Error initializing text generation"),
        })
    }
}

#[tonic::async_trait]
impl llm_server::Llm for LlmServer {
    type promptStream = ResponseStream;

    async fn prompt(&self, req: Request<PromptRequest>) -> LlmResult<Self::promptStream> {
        tracing::info!(
            "LlmServer::prompt client connected from: {:?}",
            req.remote_addr()
        );
        let (prompt_sender, mut prompt_receiver) = mpsc::channel(128);
        let (sender, receiver) = mpsc::channel(128);
        tokio::spawn(async move {
            while let Some(item) = prompt_receiver.recv().await {
                match sender.send(Result::<_, Status>::Ok(item)).await {
                    Ok(_) => {
                        // item (server response) was queued to be send to client
                    }
                    Err(_item) => {
                        // output_stream was build from receiver and both are dropped
                        break;
                    }
                }
            }
            tracing::info!("\tclient disconnected");
        });

        let generation_request = PromptGenerationRequest {
            prompt_sender,
            request: req.into_inner(),
        };
        self.generator
            .input_channel
            .send(generation_request)
            .await
            .expect("Error sending to model");

        let output_stream = ReceiverStream::new(receiver);
        Ok(Response::new(Box::pin(output_stream) as Self::promptStream))
    }
}

pub fn service() -> llm_server::LlmServer<LlmServer> {
    tracing::info!("Adding llm service");
    let server =
        LlmServer::new(ModelType::Mistral7bInstructV02).expect("Error loading llm service");
    llm_server::LlmServer::new(server)
}

use llm::{ModelType, TextGeneration};

struct PromptGenerationRequest {
    pub request: PromptRequest,
    pub prompt_sender: mpsc::Sender<PromptReply>,
}

impl From<PromptRequest> for llm::Prompt {
    fn from(value: PromptRequest) -> Self {
        Self {
            id: value.id,
            content: value.content,
        }
    }
}

impl From<llm::Prompt> for PromptRequest {
    fn from(value: llm::Prompt) -> Self {
        Self {
            id: value.id,
            content: value.content,
            config: None,
        }
    }
}

#[derive(Debug)]
struct TextGenerator {
    input_channel: mpsc::Sender<PromptGenerationRequest>,
}

impl TextGenerator {
    pub fn new(model_type: ModelType) -> crate::Result<Self> {
        let (input_channel, mut receiver) = mpsc::channel(128);

        tokio::task::spawn_blocking(move || {
            let mut text_generation =
                TextGeneration::new(model_type).expect("loading text generator");
            let mut output_channels = HashMap::new();
            tracing::info!("Model {:?} initialized.", model_type);
            let mut batch: Option<llm::BatchEncoding> = None;
            let token_chunk_length = 2;
            let mut current_token = 0;
            loop {
                if !receiver.is_empty() {
                    tracing::debug!("Picking up model input");
                    let prompt: PromptGenerationRequest = receiver
                        .blocking_recv()
                        .expect("Input channel to model closed");
                    output_channels.insert(prompt.request.id.clone(), prompt.prompt_sender);
                    let new_batch = text_generation
                        .tokenizer
                        .encode_batch(vec![prompt.request.into()], true)
                        .expect("Error encoding batch");
                    match batch {
                        Some(ref mut current_batch) => current_batch
                            .merge_batch(new_batch)
                            .expect("Error merging batches"),
                        None => batch = Some(new_batch),
                    }
                } else if let Some(mut next_batch) = batch {
                    current_token += 1;
                    if current_token == token_chunk_length {
                        tracing::debug!("Generating batch: {:?}", &next_batch);
                    }
                    let next_tokens = text_generation.next(&next_batch).expect("Error generating");
                    let is_end_of_sequence: Vec<u8> = next_tokens
                        .eq(text_generation.tokenizer.eos_id)
                        .expect("Tensor error")
                        .squeeze(1)
                        .expect("Tensor error")
                        .to_vec1()
                        .expect("Tensor error");
                    next_batch
                        .append_tokens(&next_tokens)
                        .expect("Error appending tokens");

                    let next_tokens_vec: Vec<Vec<u32>> =
                        next_batch.ids.to_vec2().expect("Tensor error");
                    let decoded = text_generation
                        .tokenizer
                        .batch_decode(&next_tokens_vec, false)
                        .expect("Error decoding tokens");
                    for (index, decoded_tokens) in decoded.iter().enumerate() {
                        let is_end_of_sequence = is_end_of_sequence[index] == 1;
                        if token_chunk_length == current_token || is_end_of_sequence {
                            let output_channel = output_channels.get(&next_batch.keys[index]);
                            if let Some(channel) = output_channel {
                                channel
                                    .blocking_send(PromptReply {
                                        id: next_batch.keys[index].clone(),
                                        content: decoded_tokens.to_owned(),
                                        meta: None,
                                        config: None,
                                        is_end_of_sequence,
                                    })
                                    .expect("Error sending PromptReply")
                            }
                        }
                        if is_end_of_sequence {
                            output_channels
                                .remove(&next_batch.keys[index])
                                .expect("Error removing channel");
                        }
                    }
                    if current_token == token_chunk_length {
                        current_token = 0;
                    }
                    if output_channels.keys().len() > 0 {
                        batch = Some(next_batch)
                    } else {
                        batch = None
                    }
                } else {
                    tracing::debug!("Waiting for model input");
                    let prompt: PromptGenerationRequest = receiver
                        .blocking_recv()
                        .expect("Input channel to model closed");
                    output_channels.insert(prompt.request.id.clone(), prompt.prompt_sender);
                    let new_batch = text_generation
                        .tokenizer
                        .encode_batch(vec![prompt.request.into()], true)
                        .expect("Error encoding batch");
                    batch = Some(new_batch)
                }
            }
        });
        Ok(Self { input_channel })
    }
}
