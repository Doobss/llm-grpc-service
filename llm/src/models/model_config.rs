#[derive(Debug, Default, Clone)]
pub struct ModelConfig {
    pub seed: u64,
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl ModelConfig {}
