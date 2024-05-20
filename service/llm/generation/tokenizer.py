import torch as T
from logging import getLogger
from transformers import BatchEncoding
from typing import List
from service.typed import (
    IntTensor,
    device,
    TransformerTokenizer
)
logger = getLogger()


class Tokenizer:

    _tokenizer: TransformerTokenizer
    _is_cuda: bool
    encoded_eos_token: IntTensor
    encoded_bos_token: IntTensor

    def __init__(self, tokenizer: TransformerTokenizer):
        self._tokenizer = tokenizer
        self._is_cuda = T.cuda.is_available()
        self.encoded_eos_token = self.tokenize([self._tokenizer.eos_token]).input_ids[0, -1]
        self.encoded_bos_token = self.tokenize([self._tokenizer.bos_token]).input_ids[0, -1]

    def tokenize(self, strings: List[str]) -> BatchEncoding:
        return self._tokenizer(
            strings,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

    def batch_decode(self, generation_output: IntTensor) -> List[str]:
        return self._tokenizer.batch_decode(
            generation_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


if __name__ == '__main__':
    ...