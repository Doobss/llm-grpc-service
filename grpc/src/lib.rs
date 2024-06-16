extern crate clap;
extern crate llm;
extern crate rand;
extern crate thiserror;
extern crate tonic;
extern crate tracing;
extern crate tracing_subscriber;
extern crate uuid;

mod error;
pub mod logging;
pub mod utils;
pub mod v1;

pub type EndpointResult<T> = std::result::Result<tonic::Response<T>, tonic::Status>;
pub type EndpointStream<T> = std::pin::Pin<
    Box<dyn tokio_stream::Stream<Item = std::result::Result<T, tonic::Status>> + Send>,
>;

pub use error::{Error, Result};
