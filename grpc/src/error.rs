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
}
