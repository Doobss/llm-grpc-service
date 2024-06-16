pub type TemplateResult<T> = core::result::Result<T, TemplateError>;

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error(transparent)]
    JninjaError(#[from] minijinja::Error),
}
