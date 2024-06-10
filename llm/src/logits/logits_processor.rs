use std::{any::Any, fmt::Debug};

use crate::Result;
use candle_core::{IndexOp, Tensor};
use candle_transformers::generation::{self, Sampling};

#[derive()]
pub struct LogitsProcessor {
    inner: generation::LogitsProcessor,
}

impl LogitsProcessor {
    pub fn sample(&mut self, logits: &Tensor) -> Result<Tensor> {
        let batch_length = logits.shape().dims()[0];
        let mut samples = Vec::new();
        for index in 0..batch_length {
            let logit_row = logits.i(index)?;
            let logit = self.inner.sample(&logit_row)?;
            samples.push(vec![logit])
        }
        Ok(Tensor::new(samples, logits.device())?)
    }
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        tracing::debug!(
            "Creating logits processor with {:?} sampling and a seed value {}",
            &sampling,
            &seed
        );
        let inner = generation::LogitsProcessor::from_sampling(seed, sampling);
        Self { inner }
    }

    pub fn from_config(config: &crate::PromptConfig) -> Self {
        let sampling = crate::utils::get_sampling(config.temperature, config.top_k, config.top_p);
        tracing::debug!(
            "Creating logits processor with {:?} sampling and a seed value {}",
            &sampling,
            &config.seed
        );

        let inner = generation::LogitsProcessor::from_sampling(config.seed, sampling);
        Self { inner }
    }
}

impl Default for LogitsProcessor {
    fn default() -> Self {
        let seed = 100;
        let sampling = generation::Sampling::ArgMax;
        let inner = generation::LogitsProcessor::from_sampling(seed, sampling);
        Self { inner }
    }
}

impl Debug for LogitsProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner_type = self.inner.type_id();
        f.debug_struct("LogitsProcessor")
            .field("inner", &inner_type)
            .finish()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx;
    use rand::{self, Rng, SeedableRng};

    struct Setup {
        pub temperature: f64,
        pub seed: u64,
        pub top_p: f64,
        pub top_k: usize,
        pub device: candle_core::Device,
        // pub dtype: candle_core::DType,
    }

    impl Setup {
        fn new() -> Result<Self> {
            Ok(Self {
                temperature: 0.5,
                seed: 100,
                top_k: 4,
                top_p: 0.5,
                device: candle_core::Device::cuda_if_available(0)?,
                // dtype: candle_core::DType::F32,
            })
        }

        fn get_logit_batch(&self, batch_size: Option<usize>) -> Result<(Tensor, usize)> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
            let batch_size = batch_size.unwrap_or(2);
            let vocab_length = 3200;
            let mut batch: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
            for _i in 0..batch_size {
                let logits: Vec<f32> = (0..vocab_length).map(|_| rng.gen_range(0.0..1.0)).collect();
                let norm = logits.iter().fold(0., |sum, &num| sum + num);
                batch.push(logits.iter().map(|&b| b / norm).collect());
            }
            Ok((Tensor::new(batch, &self.device)?, batch_size))
        }
    }

    #[test]
    fn create_logit_batch() -> Result<()> {
        let setup = Setup::new()?;
        for batch_size in 1..5 {
            let (logit_batch, _) = setup.get_logit_batch(Some(batch_size))?;
            let batch_sum: f32 = logit_batch.sum((0, 1))?.to_scalar()?;
            println!("batch_sum: {}", &batch_sum);
            let _ = approx::relative_eq!(batch_size as f32, batch_sum, epsilon = f32::EPSILON);
        }
        Ok(())
    }

    // #[test]
    // fn apply_temperature_to_logit_batch() -> Result<()> {
    //     let setup = Setup::new()?;
    //     let para_logits_processor = LogitsProcessor::from_sampling(
    //         setup.seed,
    //         generation::Sampling::All {
    //             temperature: setup.temperature,
    //         },
    //     );
    //     for batch_size in 1..5 {
    //         let (logit_batch, _) = setup.get_logit_batch(Some(batch_size))?;
    //         let tempered_batch =
    //             para_logits_processor.apply_temperature(&logit_batch, setup.temperature)?;
    //         let tempered_batch_sum: f32 = tempered_batch.sum((0, 1))?.to_scalar()?;
    //         println!("tempered_batch_sum: {}", &tempered_batch_sum);
    //         let _ = approx::relative_eq!(
    //             batch_size as f32,
    //             tempered_batch_sum,
    //             epsilon = f32::EPSILON
    //         );
    //     }
    //     Ok(())
    // }

    fn process_logits(
        logit_batch: &Tensor,
        mut lib_logits_processor: generation::LogitsProcessor,
        mut para_logits_processor: LogitsProcessor,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let batch_size = logit_batch.dim(0)?;
        let mut lib_logits = Vec::with_capacity(batch_size);
        let start_lib_sampling = std::time::Instant::now();
        for index in 0..batch_size {
            let logit = lib_logits_processor.sample(&logit_batch.i(index)?)?;
            lib_logits.push(logit);
        }
        let lib_process_time = start_lib_sampling.elapsed();
        let start_para_sampling = std::time::Instant::now();
        let para_logits = para_logits_processor
            .sample(logit_batch)?
            .squeeze(1)?
            .to_vec1::<u32>()?;
        let para_process_time = start_para_sampling.elapsed();
        println!(
            "lib_process_time: {} nanoseconds.",
            lib_process_time.as_nanos()
        );
        println!(
            "para_process_time: {} nanoseconds.",
            para_process_time.as_nanos()
        );
        println!("lib_logits: {:?}", &lib_logits);
        println!("para_logits: {:?}", &para_logits);
        Ok((lib_logits, para_logits))
    }

    #[test]
    fn sampling_arg_max() -> Result<()> {
        let setup = Setup::new()?;
        let sampling = generation::Sampling::ArgMax;
        let (logit_batch, _) = setup.get_logit_batch(Some(5))?;

        let lib_logits_processor =
            generation::LogitsProcessor::from_sampling(setup.seed, sampling.clone());
        let para_logits_processor = LogitsProcessor::from_sampling(setup.seed, sampling);
        let (lib_logits, para_logits) =
            process_logits(&logit_batch, lib_logits_processor, para_logits_processor)?;
        assert_eq!(lib_logits, para_logits);
        Ok(())
    }

    #[test]
    fn sampling_all() -> Result<()> {
        let setup = Setup::new()?;
        let sampling = generation::Sampling::All {
            temperature: setup.temperature,
        };
        let (logit_batch, _) = setup.get_logit_batch(Some(5))?;
        let lib_logits_processor =
            generation::LogitsProcessor::from_sampling(setup.seed, sampling.clone());
        let para_logits_processor = LogitsProcessor::from_sampling(setup.seed, sampling);
        let (lib_logits, para_logits) =
            process_logits(&logit_batch, lib_logits_processor, para_logits_processor)?;
        assert_eq!(lib_logits, para_logits);
        Ok(())
    }

    #[test]
    fn sampling_top_p() -> Result<()> {
        let setup = Setup::new()?;
        let sampling = generation::Sampling::TopP {
            p: setup.top_p,
            temperature: setup.temperature,
        };
        let (logit_batch, _) = setup.get_logit_batch(Some(5))?;
        let lib_logits_processor =
            generation::LogitsProcessor::from_sampling(setup.seed, sampling.clone());
        let para_logits_processor = LogitsProcessor::from_sampling(setup.seed, sampling);
        let (lib_logits, para_logits) =
            process_logits(&logit_batch, lib_logits_processor, para_logits_processor)?;
        assert_eq!(lib_logits, para_logits);
        Ok(())
    }

    #[test]
    fn sampling_top_k() -> Result<()> {
        let setup = Setup::new()?;
        let sampling = generation::Sampling::TopK {
            k: setup.top_k,
            temperature: setup.temperature,
        };
        let (logit_batch, _) = setup.get_logit_batch(Some(5))?;
        let lib_logits_processor =
            generation::LogitsProcessor::from_sampling(setup.seed, sampling.clone());
        let para_logits_processor = LogitsProcessor::from_sampling(setup.seed, sampling);
        let (lib_logits, para_logits) =
            process_logits(&logit_batch, lib_logits_processor, para_logits_processor)?;
        assert_eq!(lib_logits, para_logits);
        Ok(())
    }

    #[test]
    fn sampling_top_k_then_top_p() -> Result<()> {
        let setup = Setup::new()?;
        let sampling = generation::Sampling::TopKThenTopP {
            k: setup.top_k,
            p: setup.top_p,
            temperature: setup.temperature,
        };
        let (logit_batch, _) = setup.get_logit_batch(Some(5))?;
        let lib_logits_processor =
            generation::LogitsProcessor::from_sampling(setup.seed, sampling.clone());
        let para_logits_processor = LogitsProcessor::from_sampling(setup.seed, sampling);
        let (lib_logits, para_logits) =
            process_logits(&logit_batch, lib_logits_processor, para_logits_processor)?;
        assert_eq!(lib_logits, para_logits);
        Ok(())
    }
}
