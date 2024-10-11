use super::ModelResult;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: super::ModelType,
    pub dtype: candle_core::DType,
    pub quantize: bool,
}

impl ModelConfig {
    pub fn api_repo(&self) -> ModelResult<hf_hub::api::sync::ApiRepo> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir("./.cache/huggingface".into())
            .with_token(Some(std::env!("HUGGING_FACE_TOKEN").to_owned()))
            .build()?;
        let model_id = self.model_id.path();
        tracing::debug!("loading model_id: {model_id}");
        Ok(api.repo(hf_hub::Repo::with_revision(
            model_id,
            hf_hub::RepoType::Model,
            "main".to_owned(),
        )))
    }
}

pub fn str_to_dtype(value: &str) -> candle_core::DType {
    match value {
        "BF16" => candle_core::DType::BF16,
        "F32" => candle_core::DType::F32,
        "F16" => candle_core::DType::F16,
        "I64" => candle_core::DType::I64,
        "U32" => candle_core::DType::U32,
        "U8" => candle_core::DType::U8,
        "bf16" => candle_core::DType::BF16,
        "f32" => candle_core::DType::F32,
        "f16" => candle_core::DType::F16,
        "i64" => candle_core::DType::I64,
        "u32" => candle_core::DType::U32,
        "u8" => candle_core::DType::U8,
        _ => candle_core::DType::BF16,
    }
}
