#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelType {
    #[value(name = "7b-v0.2")]
    Mistral7bV02,
    #[value(name = "7b-instruct-v0.2")]
    Mistral7bInstructV02,
    #[value(name = "7B-v0.1")]
    Mistral7bV01,
    #[value(name = "7B-v0.1-quant")]
    QuantizedMistral7bV01,
}

impl ModelType {
    pub fn path(&self) -> String {
        match self {
            ModelType::Mistral7bV01 => "mistralai/Mistral-7B-v0.1".to_string(),
            ModelType::QuantizedMistral7bV01 => "mistralai/Mistral-7B-v0.1".to_string(),
            ModelType::Mistral7bV02 => "mistralai/Mistral-7B-v0.2".to_string(),
            ModelType::Mistral7bInstructV02 => "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
        }
    }

    pub fn id(&self) -> String {
        self.path()
    }

    pub fn is_quantized(self) -> bool {
        self == ModelType::QuantizedMistral7bV01
    }
}
