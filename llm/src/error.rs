use candle_core::error::Error as CandleError;
use hf_hub::api::sync::ApiError as HfApiError;
use huggingface_tokenizers::Error as TokenizerError;
use serde_json::Error as JsonError;
use std::io::Error as StdIoError;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    CandleError(#[from] CandleError),
    #[error(transparent)]
    HfApiError(#[from] HfApiError),
    #[error(transparent)]
    StdIoError(#[from] StdIoError),
    #[error(transparent)]
    JsonError(#[from] JsonError),
    #[error(transparent)]
    TokenizerError(#[from] TokenizerError),
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
