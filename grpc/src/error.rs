use clap::error;
use llm::Error as LlmError;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    LlmError(#[from] LlmError),
    #[error(transparent)]
    AddrParseError(#[from] std::net::AddrParseError),
    #[error(transparent)]
    TonicTransportError(#[from] tonic::transport::Error),
    #[error(transparent)]
    TonicReflectionError(#[from] tonic_reflection::server::Error),
    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl From<Error> for tonic::Status {
    fn from(value: Error) -> Self {
        tonic::Status::internal(value.to_string())
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(value: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Error::InternalError {
            message: value.to_string(),
        }
    }
}
