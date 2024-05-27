pub mod pb {
    tonic::include_proto!("llm.service");
}
use grpc::{logging, services};
use std::net::ToSocketAddrs;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init();
    tracing::info!("Starting server");
    Server::builder()
        .add_service(services::spec_service()?)
        .add_service(services::llm::service())
        .serve("[::]:50051".to_socket_addrs().unwrap().next().unwrap())
        .await
        .unwrap();
    Ok(())
}
