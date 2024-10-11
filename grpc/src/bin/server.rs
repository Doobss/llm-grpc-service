use grpc::{logging, v1};
use std::net::ToSocketAddrs;
use tonic::transport::Server;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    // /// The temperature used to generate samples.
    // #[arg(long)]
    // temperature: Option<f64>,

    // /// Nucleus sampling probability cutoff.
    // #[arg(long)]
    // top_p: Option<f64>,

    // /// Only sample among the top K samples.
    // #[arg(long)]
    // top_k: Option<usize>,

    // /// The seed to use when generating random samples.
    // #[arg(long, default_value_t = 299792458)]
    // seed: u64,
    /// The model id to use.
    #[arg(long, default_value = "7b-instruct-v0.2")]
    pub model_id: llm::ModelType,

    /// The data type to load the model in.
    #[arg(long, default_value = "BF16")]
    pub dtype: String,

    /// The data type to load the model in.
    #[arg(long, default_value = "false", default_value_t = false)]
    pub quantize: bool,

    #[arg(long, default_value = "main")]
    pub revision: String,
}

impl From<Args> for llm::ModelConfig {
    fn from(value: Args) -> Self {
        Self {
            model_id: value.model_id,
            dtype: llm::str_to_dtype(&value.dtype),
            quantize: value.quantize,
        }
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config: llm::ModelConfig = args.into();
    logging::init();
    tracing::info!("Starting server with config: {:?}", &config);
    Server::builder()
        .add_service(v1::services::spec_service()?)
        .add_service(v1::services::prompt::service(config.model_id))
        .add_service(v1::services::llm::service(config).await)
        .serve("[::]:50051".to_socket_addrs().unwrap().next().unwrap())
        .await
        .unwrap();
    Ok(())
}
