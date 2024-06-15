mod batching;
mod decoder;
mod generation;
mod tokenize;

use crate::Result;

pub type TaskResult<T> = tokio::task::JoinHandle<Result<T>>;
pub type Receiver<T> = tokio::sync::mpsc::Receiver<T>;
pub type Sender<T> = tokio::sync::mpsc::Sender<T>;
pub use batching::Batching;
pub use decoder::Decoder;
pub use generation::Generation;
pub use tokenize::Tokenize;
