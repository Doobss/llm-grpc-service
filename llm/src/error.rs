use candle_core::error::Error as CandleError;
use hf_hub::api::sync::ApiError as HfApiError;

pub type Result<T> = core::result::Result<T, Error>;

pub type ErrorMessage = String;

#[derive(Debug)]
pub enum Error {
    CandleError(ErrorMessage),
    HfApiError(ErrorMessage),
}

impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl From<CandleError> for Error {
    fn from(value: CandleError) -> Self {
        Self::CandleError(value.to_string())
    }
}

impl From<HfApiError> for Error {
    fn from(value: HfApiError) -> Self {
        Self::HfApiError(value.to_string())
    }
}

impl std::error::Error for Error {}
