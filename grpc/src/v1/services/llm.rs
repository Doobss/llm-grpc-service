use crate::v1::pb::v1::llm::service::*;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{Request, Response, Status};

type LlmResult<T> = Result<Response<T>, Status>;
type ResponseStream = Pin<Box<dyn Stream<Item = Result<PromptReply, Status>> + Send>>;

#[derive(Debug)]
pub struct LlmServer {
    generator: llm::Generator,
}

impl LlmServer {
    pub async fn new(model_type: llm::ModelType) -> crate::Result<Self> {
        Ok(Self {
            generator: llm::Generator::from_model_type(model_type)
                .await
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

        match self.generator.prompt(req.into_inner().into()).await {
            Err(error) => {
                let error: crate::Error = error.into();
                return Err(error.into());
            }
            Ok(mut result_receiver) => {
                let (prompt_sender, prompt_receiver) = mpsc::channel(128);
                let start_generation = std::time::Instant::now();
                tokio::spawn(async move {
                    while let Some(item) = result_receiver.recv().await {
                        let item: PromptReply = item.into();
                        match prompt_sender.send(Result::<_, Status>::Ok(item)).await {
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

                let output_stream = ReceiverStream::new(prompt_receiver);
                Ok(Response::new(Box::pin(output_stream) as Self::promptStream))
            }
        }
    }
}

pub async fn service(model_type: llm::ModelType) -> llm_server::LlmServer<LlmServer> {
    tracing::info!("Adding llm service");
    let server = LlmServer::new(model_type)
        .await
        .expect("Error loading llm service");
    llm_server::LlmServer::new(server)
}

impl From<llm::GenerationResult> for PromptReply {
    fn from(value: llm::GenerationResult) -> Self {
        let llm::GenerationResult {
            id,
            is_end_of_sequence,
            config,
            content,
            generated,
        } = value;
        Self {
            id,
            content,
            is_end_of_sequence,
            config: Some(config.into()),
            meta: None,
            generated,
        }
    }
}

impl From<llm::PromptConfig> for PromptConfig {
    fn from(value: llm::PromptConfig) -> Self {
        let llm::PromptConfig {
            max_new_tokens,
            num_beams,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
        } = value;
        Self {
            max_new_tokens: max_new_tokens.unwrap_or_default(),
            num_beams: num_beams.unwrap_or_default(),
            temperature: temperature.unwrap_or_default() as f32,
            top_k: top_k.unwrap_or_default() as i32,
            top_p: top_p.unwrap_or_default() as f32,
            repetition_penalty: repetition_penalty.unwrap_or_default(),
            seed: seed as i64,
        }
    }
}
