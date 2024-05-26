pub mod pb {
    tonic::include_proto!("llm.service");
}
use pb::{PromptReply, PromptRequest};
use std::{pin::Pin, time::Duration};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status};

type LlmResult<T> = Result<Response<T>, Status>;
type ResponseStream = Pin<Box<dyn Stream<Item = Result<PromptReply, Status>> + Send>>;

#[derive(Debug)]
pub struct LlmServer {}

#[tonic::async_trait]
impl pb::llm_server::Llm for LlmServer {
    type promptStream = ResponseStream;

    async fn prompt(&self, req: Request<PromptRequest>) -> LlmResult<Self::promptStream> {
        tracing::info!("LlmServer::prompt");
        tracing::info!("\tclient connected from: {:?}", req.remote_addr());

        // creating infinite stream with requested message
        let repeat = std::iter::repeat(PromptReply {
            content: "hello".to_owned(),
            id: "test".to_owned(),
            is_end_of_sequence: false,
            config: None,
            meta: None,
        });
        let mut stream = Box::pin(tokio_stream::iter(repeat).throttle(Duration::from_millis(200)));

        // spawn and channel are required if you want handle "disconnect" functionality
        // the `out_stream` will not be polled after client disconnect
        let (tx, rx) = mpsc::channel(128);
        tokio::spawn(async move {
            while let Some(item) = stream.next().await {
                match tx.send(Result::<_, Status>::Ok(item)).await {
                    Ok(_) => {
                        // item (server response) was queued to be send to client
                    }
                    Err(_item) => {
                        // output_stream was build from rx and both are dropped
                        break;
                    }
                }
            }
            tracing::info!("\tclient disconnected");
        });

        let output_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::promptStream))
    }
}

pub fn service() -> pb::llm_server::LlmServer<LlmServer> {
    tracing::info!("Adding llm service");
    let server = LlmServer {};
    pb::llm_server::LlmServer::new(server)
}
