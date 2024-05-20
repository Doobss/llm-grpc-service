from dataclasses import dataclass


@dataclass
class GenerationPromptResult:
    id: str
    prompt: str
    generated: str
    batch_size: int
    reached_eos: bool
    process_time: float
