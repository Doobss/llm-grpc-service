import logging
from pathlib import Path
from typing import Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from .modelConfig import ModelConfig
from service.typed import TransformerTokenizer, TransformerModel
logger = logging.getLogger()
ROOT_PATH = Path(__file__).parent.parent


def load_model_tokenizer_for_generate(pretrained_model_name_or_path: str, config: ModelConfig) -> Tuple[TransformerModel, TransformerTokenizer]:
    logger.info(f"Loading model and tokenizer for {pretrained_model_name_or_path}")
    tokenizer: TransformerTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        padding_side="left",
        quantization_config=config.quantization,
    )
    tokenizer.pad_token = tokenizer.eos_token \
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None \
        else tokenizer.pad_token
    model: TransformerModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        # device_map="auto",
        trust_remote_code=True,
        quantization_config=config.quantization,
        torch_dtype=config.dtype,
        max_memory={
            'cpu': '0GB'
        }
    )
    logger.info(f'{model.dtype = }')
    logger.info(f'{model.device = }')
    logger.info(f'{model.is_parallelizable = }')
    logger.info(f'{model.num_parameters() = }')
    for i in model.named_parameters():
        logger.debug(f"model named_param: {i[0]} -> {i[1].device}")
    # model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

