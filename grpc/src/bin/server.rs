use grpc::{logging, v1};
use std::net::ToSocketAddrs;
use tonic::transport::Server;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The model size to use.
    #[arg(long, default_value = "7b-instruct-v0.2")]
    model: llm::ModelType,

    #[arg(long, default_value = "main")]
    revision: String,
    // #[arg(long)]
    // quantized: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    logging::init();
    tracing::info!("Starting server");
    Server::builder()
        .add_service(v1::services::spec_service()?)
        .add_service(v1::services::llm::service(args.model))
        .serve("[::]:50051".to_socket_addrs().unwrap().next().unwrap())
        .await
        .unwrap();
    Ok(())
}
