pub type ModelResult<T> = core::result::Result<T, ModelError>;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
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
    #[error("Generation error: {message}")]
    GenerationError { message: String },
}
