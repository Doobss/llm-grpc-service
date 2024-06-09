#[derive(Debug, Default)]
pub struct PromptConfig {
    pub max_new_tokens: i32,
    pub num_beams: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}
