from dataclasses import dataclass


@dataclass
class GenerationChunk:
    generated: str
    is_end_of_sequence: bool

