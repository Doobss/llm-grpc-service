from typing import Dict, Optional, TypeVar, Generic, Tuple, List, Generator, Union


T = TypeVar('T')


class GenerationBatch(Generic[T]):
    _batch: Dict[str, T]

    def __init__(self, prompts: Union[Dict[str, T], List[T]] = None) -> None:
        self._batch = {}
        if isinstance(prompts, dict):
            self._batch = prompts or {}
        elif isinstance(prompts, list):
            self._batch = {
                prompt.id: prompt
                for prompt in prompts
            }

    def __setitem__(self, key, value: T) -> None:
        self._batch[key] = value

    def __getitem__(self, key) -> Optional[T]:
        return self._batch.get(key, None)

    def __len__(self) -> int:
        return len(self._batch.keys())

    def __contains__(self, item: str) -> bool:
        if isinstance(item, str):
            return item in self._batch
        return False

    def __iter__(self) -> Generator[Tuple[str, T], None, None]:
        return (x for x in self._batch.items())

    def __delitem__(self, key) -> None:
        if isinstance(key, str):
            if key in self._batch:
                del self._batch[key]