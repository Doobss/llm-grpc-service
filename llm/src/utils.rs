use crate::*;
use candle_core::Tensor;

pub fn get_eos_tokens(tokens: &Tensor, eos_id: u32) -> Result<Vec<u8>> {
    Ok(tokens.eq(eos_id)?.squeeze(1)?.to_vec1::<u8>()?)
}

pub fn load_file<T>(file_path: std::path::PathBuf) -> Result<T>
where
    T: for<'a> serde::Deserialize<'a>,
{
    let file_data = std::fs::read(file_path)?;
    let value: T = serde_json::from_slice(&file_data)?;
    Ok(value)
}

pub fn get_sampling(
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> crate::Sampling {
    let temperature = temperature.unwrap_or_default();
    if temperature <= 0. {
        crate::Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => crate::Sampling::All { temperature },
            (Some(k), None) => crate::Sampling::TopK { k, temperature },
            (None, Some(p)) => crate::Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => crate::Sampling::TopKThenTopP { k, p, temperature },
        }
    }
}
