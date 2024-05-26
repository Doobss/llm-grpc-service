use candle_core::error::Error as CandleError;
use hf_hub::api::sync::ApiError as HfApiError;
use serde_json::Error as JsonError;
use std::io::Error as StdIoError;
use tokenizers::Error as TokenizerError;

pub type Result<T> = core::result::Result<T, Error>;

pub type ErrorMessage = String;

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
}
