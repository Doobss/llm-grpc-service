#[derive(Debug, Default, Clone)]
pub struct PromptConfig {
    pub max_new_tokens: Option<i32>,
    pub num_beams: Option<i32>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub seed: u64,
}
