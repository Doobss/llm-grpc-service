extern crate candle_core;

use candle_core::error::Error as CandleError;

pub type Result<T> = core::result::Result<T, Error>;

pub type ErrorMessage = String;

#[derive(Debug)]
pub enum Error {
    CandleError(ErrorMessage),
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

impl std::error::Error for Error {}
