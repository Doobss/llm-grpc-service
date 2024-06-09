use crate::v1::pb::v1::llm::service::*;
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
    pub fn new(model_type: ModelType, sampling: llm::Sampling) -> crate::Result<Self> {
        Ok(Self {
            generator: TextGenerator::new(model_type, sampling)
                .expect("Error initializing text generation"),
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
        let start_generation = std::time::Instant::now();
        tokio::spawn(async move {
            while let Some(item) = prompt_receiver.recv().await {
                match sender.send(Result::<_, Status>::Ok(item)).await {
                    Ok(_) => (),
                    Err(_item) => {
                        // output_stream was build from receiver and both are dropped
                        break;
                    }
                }
            }
            let generation_duration = start_generation.elapsed();
            tracing::info!(
                "client disconnected generation_duration: {:?} ms",
                generation_duration.as_millis()
            );
        });

        let generation_request = PromptGenerationRequest {
            prompt_sender,
            request: req.into_inner().into(),
        };
        let result = self.generator.input_channel.send(generation_request).await;
        if let Err(error) = result {
            let error: crate::Error = error.into();
            return Err(error.into());
        }

        let output_stream = ReceiverStream::new(receiver);
        Ok(Response::new(Box::pin(output_stream) as Self::promptStream))
    }
}

pub fn service(model_type: ModelType, sampling: llm::Sampling) -> llm_server::LlmServer<LlmServer> {
    tracing::info!("Adding llm service");
    let server = LlmServer::new(model_type, sampling).expect("Error loading llm service");
    llm_server::LlmServer::new(server)
}

use llm::{ModelType, Prompt, TextGeneration};

struct PromptGenerationRequest {
    pub request: Prompt,
    pub prompt_sender: mpsc::Sender<PromptReply>,
}

impl From<PromptConfig> for llm::PromptConfig {
    fn from(value: PromptConfig) -> Self {
        Self {
            max_new_tokens: value.max_new_tokens,
            num_beams: value.num_beams,
            temperature: value.temperature,
            top_k: value.top_k,
            top_p: value.top_p,
            repetition_penalty: value.repetition_penalty,
        }
    }
}

impl From<PromptRequest> for llm::Prompt {
    fn from(value: PromptRequest) -> Self {
        let id = if value.id.is_empty() {
            llm::Prompt::gen_id()
        } else {
            value.id
        };
        let config = if let Some(config) = value.config {
            config.into()
        } else {
            llm::PromptConfig::default()
        };
        tracing::info!("Prompt {:?} config: {:?}", &id, &config);
        Self {
            id,
            content: value.content,
            config,
        }
    }
}

#[derive(Debug)]
struct TextGenerator {
    input_channel: mpsc::Sender<PromptGenerationRequest>,
}

impl TextGenerator {
    pub fn new(model_type: ModelType, sampling: llm::Sampling) -> crate::Result<Self> {
        let (input_channel, mut receiver) = mpsc::channel(128);

        tokio::task::spawn_blocking(move || {
            let mut text_generation =
                TextGeneration::new(model_type, Some(sampling)).expect("loading text generator");
            let mut output_channels = HashMap::new();
            tracing::info!("Model {:?} initialized.", model_type);
            let mut batch: Option<llm::BatchEncoding> = None;
            let token_chunk_length = 2;
            let mut current_token = 0;
            loop {
                if !receiver.is_empty() {
                    tracing::debug!("Picking up model input");
                    let mut prompts = Vec::new();
                    while !receiver.is_empty() {
                        let prompt: PromptGenerationRequest = receiver
                            .blocking_recv()
                            .expect("Input channel to model closed");
                        output_channels.insert(prompt.request.id.clone(), prompt.prompt_sender);
                        prompts.push(prompt.request)
                    }
                    let new_batch = text_generation
                        .tokenizer
                        .encode_batch(prompts, true)
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
                    let start_generation = std::time::Instant::now();
                    let next_tokens = text_generation.next(&next_batch).expect("Error generating");
                    let generation_duration = start_generation.elapsed();

                    let start_token_decode = std::time::Instant::now();
                    let is_end_of_sequence =
                        llm::get_eos_tokens(&next_tokens, text_generation.tokenizer.eos_id)
                            .expect("Error getting eos tokens");

                    next_batch
                        .append_tokens(&next_tokens)
                        .expect("Error appending tokens");

                    let decoded = text_generation
                        .tokenizer
                        .decode_batch(&next_batch)
                        .expect("Error decoding tokens");
                    let decoding_duration = start_token_decode.elapsed();

                    let start_sending_results = std::time::Instant::now();
                    for (index, decoded_tokens) in decoded.iter().enumerate() {
                        let is_end_of_sequence = is_end_of_sequence[index] == 1;
                        if token_chunk_length == current_token || is_end_of_sequence {
                            let output_channel = output_channels.get(&next_batch.keys[index]);
                            if let Some(channel) = output_channel {
                                if !channel.is_closed() {
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
                        }
                        if is_end_of_sequence {
                            output_channels
                                .remove(&next_batch.keys[index])
                                .expect("Error removing channel");
                        }
                    }
                    let sending_results_duration = start_sending_results.elapsed();
                    tracing::info!(
                        "generation_duration: {:?} ms | decoding_duration: {:?} ms | sending_results_duration: {:?} ms",
                        generation_duration.as_millis(),
                        decoding_duration.as_millis(),
                        sending_results_duration.as_millis()
                    );

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
                        .encode_batch(vec![prompt.request], true)
                        .expect("Error encoding batch");
                    batch = Some(new_batch)
                }
            }
        });
        Ok(Self { input_channel })
    }
}
