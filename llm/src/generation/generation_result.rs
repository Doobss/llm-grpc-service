use crate::PromptConfig;

#[derive(Debug)]
pub struct GenerationResult {
    pub id: String,
    pub content: String,
    pub generated: String,
    pub is_end_of_sequence: bool,
    pub config: PromptConfig,
}

impl GenerationResult {}
