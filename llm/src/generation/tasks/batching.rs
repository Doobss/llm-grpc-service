use super::{Receiver, Sender, TaskResult};
use crate::{Error, GenerationBatch, GenerationRequest};

#[derive(Debug)]
pub struct Batching {
    request_receiver: Receiver<GenerationRequest>,
    batch_sender: Sender<GenerationBatch>,
}

impl Batching {
    pub fn task(self) -> TaskResult<()> {
        let Batching {
            batch_sender,
            mut request_receiver,
        } = self;
        tokio::task::spawn(async move {
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
        })
    }
}

impl Batching {
    pub fn new(
        request_receiver: Receiver<GenerationRequest>,
        batch_sender: Sender<GenerationBatch>,
    ) -> Self {
        Self {
            request_receiver,
            batch_sender,
        }
    }
}
