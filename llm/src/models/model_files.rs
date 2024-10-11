use super::ModelType;
use crate::ModelResult;
use candle_examples::hub_load_safetensors;
use hf_hub::api::sync::ApiRepo;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub model_type: ModelType,
    pub config: PathBuf,
    pub weights: Vec<PathBuf>,
    pub quantized_weights: Vec<PathBuf>,
    pub generation_config: Option<PathBuf>,
}

impl ModelFiles {
    pub fn from_repo(model_type: ModelType, repo: &ApiRepo) -> ModelResult<Self> {
        let config = repo.get("config.json")?;
        let generation_config = match repo.get("generation_config.json") {
            Ok(config_path) => Some(config_path),
            Err(_) => None,
        };
        let mut weights = Vec::new();
        let mut quantized_weights = Vec::new();
        let repo_info = repo.info()?;
        if model_type.is_quantizable() {
            quantized_weights = repo_info
                .siblings
                .into_iter()
                .filter_map(|sibling| {
                    let filename = sibling.rfilename.clone();
                    tracing::debug!("sibling filename: {:?}", &filename);
                    if filename.ends_with(".gguf") {
                        return repo.get(&filename).ok();
                    }
                    None
                })
                .collect();
            if quantized_weights.is_empty() {
                tracing::debug!("attempting to get model-q4k.gguf");
                quantized_weights = vec![repo.get("model-q4k.gguf")?]
            }
        } else {
            weights = hub_load_safetensors(repo, "model.safetensors.index.json")?;
        }
        Ok(Self {
            model_type,
            config,
            weights,
            quantized_weights,
            generation_config,
        })
    }

    pub fn load_file<T>(file_path: std::path::PathBuf) -> ModelResult<T>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        let file_data = std::fs::read(file_path)?;
        let value: T = serde_json::from_slice(&file_data)?;
        Ok(value)
    }

    pub fn load_config<T>(&self) -> ModelResult<T>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        ModelFiles::load_file(self.config.clone())
    }

    pub fn load_generation_config<T>(&self) -> ModelResult<Option<T>>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        if let Some(config_path) = self.generation_config.clone() {
            Ok(Some(ModelFiles::load_file(config_path)?))
        } else {
            Ok(None)
        }
    }
}
