use rand::Rng;

#[derive(Debug, Clone)]
pub struct PromptConfig {
    pub max_new_tokens: i32,
    pub num_beams: Option<i32>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub seed: u64,
}

impl Default for PromptConfig {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            max_new_tokens: 100,
            num_beams: Default::default(),
            temperature: Default::default(),
            top_k: Default::default(),
            top_p: Default::default(),
            repetition_penalty: Default::default(),
            seed: rng.gen(),
        }
    }
}
