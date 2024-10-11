#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelType {
    #[value(name = "7B-v0.1")]
    Mistral7bV01,
    #[value(name = "7b-instruct-v0.1")]
    Mistral7bInstructV01,
    #[value(name = "7b-v0.2")]
    Mistral7bV02,
    #[value(name = "7b-instruct-v0.2")]
    Mistral7bInstructV02,
    #[value(name = "7b-v0.3")]
    Mistral7bV03,
    #[value(name = "7b-instruct-v0.3")]
    Mistral7bInstructV03,
}

impl ModelType {
    pub fn path(&self) -> String {
        match self {
            ModelType::Mistral7bV01 => "mistralai/Mistral-7B-v0.1".to_string(),
            ModelType::Mistral7bInstructV01 => "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
            ModelType::Mistral7bV02 => "mistralai/Mistral-7B-v0.2".to_string(),
            ModelType::Mistral7bInstructV02 => "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
            ModelType::Mistral7bV03 => "mistralai/Mistral-7B-v0.3".to_string(),
            ModelType::Mistral7bInstructV03 => "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
        }
    }

    pub fn id(&self) -> String {
        self.path()
    }

    pub fn is_quantizable(self) -> bool {
        false
    }
}
