from typing import Dict, Optional, Union, List
from .generationBatch import GenerationBatch
from .tokenizedGenerationPrompt import TokenizedGenerationPrompt
from .generationChunk import GenerationChunk
from .generationPromptBatch import GenerationPromptBatch
from .generationPrompt import GenerationPrompt
from .generationPromprResultBatch import GenerationPromptResultBatch
from .tokenizer import Tokenizer
from transformers import BatchEncoding
from service.typed import IntTensor


class TokenizedGenerationPromptBatch(GenerationBatch[TokenizedGenerationPrompt]):
    _encoded_cache: Union[BatchEncoding, None] = None
    _tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer, prompts: Union[Dict[str, GenerationPrompt], List[GenerationPrompt]] = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._batch = {}
        if isinstance(prompts, dict):
            self._batch = {
                key: TokenizedGenerationPrompt(non_tokenized=prompt)
                for key, prompt in prompts.items()
            }
        elif isinstance(prompts, list):
            self._batch = {
                prompt.id: TokenizedGenerationPrompt(non_tokenized=prompt)
                for prompt in prompts
            }

    def __iadd__(
            self,
            other: Union['TokenizedGenerationPromptBatch', TokenizedGenerationPrompt, GenerationPromptBatch, GenerationPrompt]
    ) -> 'TokenizedGenerationPromptBatch':
        if other is not None:
            self._encoded_cache = None
            if isinstance(other, GenerationPromptBatch):
                for key, prompt in other:
                    self[key] = TokenizedGenerationPrompt(non_tokenized=prompt)
            elif isinstance(other, GenerationPrompt):
                self[other.id] = TokenizedGenerationPrompt(non_tokenized=other)
            elif isinstance(other, TokenizedGenerationPrompt):
                self[other.id] = other
            elif isinstance(other, TokenizedGenerationPromptBatch):
                for prompt in other.prompts:
                    self[prompt.id] = prompt
            else:
                raise ValueError(f'Adding to {type(self)} not implemented for type {type(other)}')
        return self

    def __setitem__(self, key, value) -> None:
        self._encoded_cache = None
        # TODO: concat encoded prompt instead of clearing cache to reduce call to tokenize
        if isinstance(value, TokenizedGenerationPrompt):
            self._batch[key] = value
        elif isinstance(value, GenerationPrompt):
            self._batch[key] = TokenizedGenerationPrompt(non_tokenized=value)
        else:
            raise ValueError(f'Setting to {type(self)} not implemented for type {type(value)}')

    def __getitem__(self, key) -> Optional[TokenizedGenerationPrompt]:
        return self._batch.get(key, None)

    def __len__(self) -> int:
        return len(self._batch.keys())

    def __contains__(self, item: str) -> bool:
        if isinstance(item, str):
            return item in self._batch
        return False

    def __repr__(self) -> str:
        prompts = str(self.prompts).replace('\n', '\n\t')
        return f'TokenizedGenerationPromptBatch(\n\tprompts={prompts}\n)'

    @property
    def finished_generation(self) -> bool:
        return all(generated.reached_eos for generated in self.prompts)

    @property
    def prompts(self) -> List[TokenizedGenerationPrompt]:
        return [prompt for prompt in self._batch.values()]

    @property
    def generated_prompts(self) -> List[TokenizedGenerationPrompt]:
        return [prompt for prompt in self.prompts if not prompt.reached_eos]

    @property
    def finished_prompts(self) -> List[TokenizedGenerationPrompt]:
        return [prompt for prompt in self.prompts if prompt.reached_eos]

    @property
    def gen_id(self) -> str:
        return '-'.join([prompt.id for prompt in self.generated_prompts])

    @property
    def model_inputs(self) -> BatchEncoding:
        if self._encoded_cache is not None:
            return self._encoded_cache
        inputs = self._tokenizer.tokenize([
            f'{prompt.prompt}{prompt.generated}' for prompt in self.generated_prompts
        ])
        self._encoded_cache = inputs
        return self._encoded_cache

    def process_chunk(self, generation_output: IntTensor) -> 'TokenizedGenerationPromptBatch':
        input_length = self.model_inputs.input_ids.shape[1]
        self._encoded_cache = None
        repeated_tokens = 2
        new_tokens = generation_output[:, input_length - repeated_tokens:]
        is_eos = self.is_end_of_sequence(new_tokens)
        decoded = self._tokenizer.batch_decode(new_tokens)
        repeated_tokens = self._tokenizer.batch_decode(new_tokens[:, :repeated_tokens])
        for index, prompt in enumerate(self.generated_prompts):
            prompt.process_chunk(GenerationChunk(
                generated=decoded[index].replace(repeated_tokens[index], ""),
                is_end_of_sequence=is_eos[index]
            ))
        return self

    def results(self, process_time: float) -> GenerationPromptResultBatch:
        batch_size = len(self)
        results = GenerationPromptResultBatch()
        results.process_time = process_time
        for prompt in self.prompts:
            results[prompt.id] = prompt.result(
                process_time=process_time,
                batch_size=batch_size
            )
            if prompt.reached_eos:
                del self[prompt.id]
        return results

    def is_end_of_sequence(self, generation_output: IntTensor) -> list[bool]:
        equal_to_eos = generation_output == self._tokenizer.encoded_eos_token
        return equal_to_eos.any(dim=1).tolist()
