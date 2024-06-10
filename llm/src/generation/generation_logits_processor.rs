use std::fmt::Debug;

use crate::{LogitsPreProcessor, LogitsProcessor, PromptConfig};

#[derive(Debug)]
pub struct GenerationLogitsProcessor {
    pub preprocess: LogitsPreProcessor,
    pub process: LogitsProcessor,
}

impl GenerationLogitsProcessor {
    pub fn from_prompt_config(config: &PromptConfig) -> Self {
        Self {
            preprocess: LogitsPreProcessor::from_config(config),
            process: LogitsProcessor::from_config(config),
        }
    }
}
