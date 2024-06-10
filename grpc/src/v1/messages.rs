use super::pb::v1::llm::service::*;
use rand::prelude::*;

fn default_to_optional<T>(value: T) -> Option<T>
where
    T: Default + PartialEq,
{
    if value == T::default() {
        None
    } else {
        Some(value)
    }
}

impl From<PromptConfig> for llm::PromptConfig {
    fn from(value: PromptConfig) -> Self {
        let seed: u64 = if let Some(value) = default_to_optional(value.seed) {
            value as u64
        } else {
            let mut rng = rand::thread_rng();
            rng.gen()
        };
        Self {
            max_new_tokens: default_to_optional(value.max_new_tokens),
            num_beams: default_to_optional(value.num_beams),
            temperature: default_to_optional(value.temperature as f64),
            top_k: default_to_optional(value.top_k as usize),
            top_p: default_to_optional(value.top_p as f64),
            repetition_penalty: default_to_optional(value.repetition_penalty),
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
        let id = default_to_optional(value.id).unwrap_or(llm::Prompt::gen_id());
        tracing::debug!("Prompt.id: {:?} config: {:?}", &id, &config);
        Self {
            id,
            content: value.content,
            config,
        }
    }
}
