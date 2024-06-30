#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: super::ModelType,
    pub dtype: candle_core::DType,
    pub quantize: bool,
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
