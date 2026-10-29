# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from omegaconf import DictConfig
from memora.builder.memory_builder import MemoryBuilder
from memora.core.memory import AgentMemory
from memora.utils.llm import ChatCompletionModel


class MemoryBuilderRegistry:
    _builders: dict[str, type[MemoryBuilder]] = {}

    @classmethod
    def register(cls, file_type: str, builder_cls: type[MemoryBuilder]):
        cls._builders[file_type] = builder_cls

    @classmethod
    def get(cls, file_type: str, cfg: DictConfig, agent_memory: AgentMemory, model_client: ChatCompletionModel) -> MemoryBuilder:
        try:
            return cls._builders[file_type](cfg, agent_memory, model_client)
        except KeyError:
            raise ValueError(f"No MemoryBuilder registered for '{file_type}'")
