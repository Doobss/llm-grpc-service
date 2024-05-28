use crate::*;
use candle_core::Tensor;

pub fn get_eos_tokens(tokens: &Tensor, eos_id: u32) -> Result<Vec<u8>> {
    Ok(tokens.eq(eos_id)?.squeeze(1)?.to_vec1::<u8>()?)
}
