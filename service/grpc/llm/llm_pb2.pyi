from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PromptConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PromptRequest(_message.Message):
    __slots__ = ("id", "content", "config")
    ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    id: str
    content: str
    config: PromptConfig
    def __init__(self, id: _Optional[str] = ..., content: _Optional[str] = ..., config: _Optional[_Union[PromptConfig, _Mapping]] = ...) -> None: ...

class PromptMetaData(_message.Message):
    __slots__ = ("tokens_per_second", "average_batch_size", "model")
    class Model(_message.Message):
        __slots__ = ("device", "dtype", "type", "number_of_devices")
        DEVICE_FIELD_NUMBER: _ClassVar[int]
        DTYPE_FIELD_NUMBER: _ClassVar[int]
        TYPE_FIELD_NUMBER: _ClassVar[int]
        NUMBER_OF_DEVICES_FIELD_NUMBER: _ClassVar[int]
        device: str
        dtype: str
        type: str
        number_of_devices: int
        def __init__(self, device: _Optional[str] = ..., dtype: _Optional[str] = ..., type: _Optional[str] = ..., number_of_devices: _Optional[int] = ...) -> None: ...
    TOKENS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    tokens_per_second: float
    average_batch_size: float
    model: PromptMetaData.Model
    def __init__(self, tokens_per_second: _Optional[float] = ..., average_batch_size: _Optional[float] = ..., model: _Optional[_Union[PromptMetaData.Model, _Mapping]] = ...) -> None: ...

class PromptReply(_message.Message):
    __slots__ = ("id", "content", "is_end_of_sequence", "config", "meta")
    ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    IS_END_OF_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    id: str
    content: str
    is_end_of_sequence: bool
    config: PromptConfig
    meta: PromptMetaData
    def __init__(self, id: _Optional[str] = ..., content: _Optional[str] = ..., is_end_of_sequence: bool = ..., config: _Optional[_Union[PromptConfig, _Mapping]] = ..., meta: _Optional[_Union[PromptMetaData, _Mapping]] = ...) -> None: ...
