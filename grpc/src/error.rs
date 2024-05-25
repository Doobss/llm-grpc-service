use llm::Error as LlmError;

pub type Result<T> = core::result::Result<T, Error>;

pub type ErrorMessage = String;

#[derive(Debug)]
pub enum Error {
    LlmError(ErrorMessage),
}

impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl From<LlmError> for Error {
    fn from(value: LlmError) -> Self {
        Self::LlmError(value.to_string())
    }
}

impl std::error::Error for Error {}
