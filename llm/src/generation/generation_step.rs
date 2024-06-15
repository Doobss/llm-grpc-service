use crate::TokenizedBatch;
use candle_core::Tensor;

#[derive(Debug)]
pub struct GenerationStep {
    pub batch: TokenizedBatch,
    pub logits: Tensor,
}
