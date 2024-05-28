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
