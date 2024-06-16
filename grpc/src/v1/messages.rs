use super::pb::v1::llm::service::*;
use crate::utils;
use rand::prelude::*;

impl From<PromptConfig> for llm::PromptConfig {
    fn from(value: PromptConfig) -> Self {
        let seed: u64 = if let Some(value) = utils::default_to_optional(value.seed) {
            value as u64
        } else {
            let mut rng = rand::thread_rng();
            rng.gen()
        };
        Self {
            max_new_tokens: utils::default_to_optional(value.max_new_tokens),
            num_beams: utils::default_to_optional(value.num_beams),
            temperature: utils::default_to_optional(value.temperature as f64),
            top_k: utils::default_to_optional(value.top_k as usize),
            top_p: utils::default_to_optional(value.top_p as f64),
            repetition_penalty: utils::default_to_optional(value.repetition_penalty),
            seed,
        }
    }
}

impl From<PromptRequest> for llm::Prompt {
    fn from(value: PromptRequest) -> Self {
        let config = if let Some(config) = value.config {
            config.into()
        } else {
            llm::PromptConfig::default()
        };
        let id = utils::default_to_optional(value.id).unwrap_or(llm::Prompt::gen_id());
        tracing::debug!("Prompt.id: {:?} config: {:?}", &id, &config);
        Self {
            id,
            content: value.content,
            config,
        }
    }
}
