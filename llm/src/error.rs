pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    CandleError(#[from] candle_core::error::Error),
    #[error(transparent)]
    HfApiError(#[from] hf_hub::api::sync::ApiError),
    #[error(transparent)]
    StdIoError(#[from] std::io::Error),
    #[error(transparent)]
    JsonError(#[from] serde_json::Error),
    #[error(transparent)]
    TokenizerError(#[from] crate::tokenizers::TokenizerError),
    #[error(transparent)]
    ModelError(#[from] crate::models::ModelError),
    #[error("Generation error: {message}")]
    GenerationError { message: String },
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(value: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Error::GenerationError {
            message: value.to_string(),
        }
    }
}

impl From<tokio::sync::mpsc::error::TryRecvError> for Error {
    fn from(value: tokio::sync::mpsc::error::TryRecvError) -> Self {
        Error::GenerationError {
            message: value.to_string(),
        }
    }
}
