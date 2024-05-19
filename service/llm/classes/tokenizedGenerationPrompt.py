import re
from .generationChunk import GenerationChunk
from .generationPrompt import GenerationPrompt
from .generationPromptResult import GenerationPromptResult


class TokenizedGenerationPrompt:
    id: str
    prompt: str
    generated: str = ""
    reached_eos: bool = False

    def __init__(self, non_tokenized: GenerationPrompt) -> None:
        self.id = non_tokenized.id
        self.prompt = non_tokenized.prompt

    def process_chunk(self, chunk: GenerationChunk) -> 'TokenizedGenerationPrompt':
        self.reached_eos = chunk.is_end_of_sequence or self.reached_eos
        self.generated += ' ' + chunk.generated if \
            should_add_space(self.generated, chunk.generated) else chunk.generated
        return self

    def result(self, process_time: float, batch_size: int) -> GenerationPromptResult:
        return GenerationPromptResult(
            id=self.id,
            prompt=self.prompt,
            batch_size=batch_size,
            generated=self.generated,
            process_time=process_time,
            reached_eos=self.reached_eos,
        )

    def __repr__(self) -> str:
        return (f'TokenizedGenerationPrompt(\n\tid={self.id},'
                f'\n\treached_eos={self.reached_eos},\n\tgenerated={self.generated},\n)')


def should_add_space(current: str, generated: str) -> bool:
    return False
    # return not ends_with_whitespace(current) and not starts_with_whitespace(generated)


def starts_with_whitespace(string: str) -> bool:
    return bool(re.match(r'^\s', string))


def ends_with_whitespace(string: str) -> bool:
    return bool(re.match(r'\s$', string))
