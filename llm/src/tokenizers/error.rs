pub type TokenizerResult<T> = core::result::Result<T, TokenizerError>;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error(transparent)]
    TemplateError(#[from] super::template::TemplateError),
    #[error(transparent)]
    CandleError(#[from] candle_core::error::Error),
    #[error(transparent)]
    HfApiError(#[from] hf_hub::api::sync::ApiError),
    #[error(transparent)]
    StdIoError(#[from] std::io::Error),
    #[error(transparent)]
    JsonError(#[from] serde_json::Error),
    #[error(transparent)]
    TokenizerError(#[from] huggingface_tokenizers::Error),
}
