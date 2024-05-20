import torch
from os import getenv
from math import isnan
from json import loads
from logging import getLogger
from transformers import (
    GenerationConfig,
    BitsAndBytesConfig,
    PretrainedConfig
)
from typing import Optional, Union
logger = getLogger()


class ModelConfig:
    quantization: Union[BitsAndBytesConfig, None]
    generation: GenerationConfig
    model: PretrainedConfig
    chunk_size: int
    max_batch_size: int
    dtype: torch.dtype

    def __init__(self):
        self.quantization = _get_default_quantization_config()
        self.generation = _get_default_generation_config()
        self.dtype = _get_dtype() or torch.bfloat16
        self.chunk_size = _get_int_env('chunk_size', 17)
        self.max_batch_size = _get_int_env('max_batch_size', 50)
        logger.info(f'ModelConfig: {self.dtype = }')

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "generation": self.generation.to_dict(),
            "quantization": self.quantization.to_dict() if self.quantization is not None else None,
            "max_batch_size": self.max_batch_size,
            "chunk_size": self.chunk_size,
            "dtype": str(self.dtype)
        }


def _get_default_quantization_config() -> BitsAndBytesConfig:
    quant_config = BitsAndBytesConfig(

    )
    logger.info(f"Loaded quantization config from consul: {quant_config}")
    return quant_config


def _get_default_generation_config() -> GenerationConfig:
    generation_config = GenerationConfig(
        temperature=0.55,
        do_sample=True,
        repetition_penalty=1.1,
        exponential_decay_length_penalty=(200, 1.5)
    )
    logger.info(f"Loaded generation config from consul: {generation_config}")
    generation_config.pad_token_id = 1 if generation_config.pad_token_id is None else generation_config.pad_token_id
    return generation_config


def _get_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    dtype_name = getenv('services/llm/dtype', default='bfloat16')
    if hasattr(torch, dtype_name) and isinstance(getattr(torch, dtype_name), torch.dtype):
        return getattr(torch, dtype_name)
    return torch.bfloat16


def _get_int_env(name: str, default: int, base: str = 'services/llm') -> int:
    chunk_size = getenv(f'{base}/{name}', default=default)
    if isinstance(chunk_size, str):
        chunk_size = int(chunk_size)
    return chunk_size if isinstance(chunk_size, int) and not isnan(chunk_size) else default