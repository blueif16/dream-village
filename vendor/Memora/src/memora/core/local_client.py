# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
MemoraClient - Public-facing facade requiring only an API key.

Changes added (brief):
 - Removed unused imports (curses.meta, regex.P, chromadb Where) to reduce noise.
 - Added os and OmegaConf imports for default config builder.
 - Corrected API key util import function name.
 - Consolidated duplicate add/query methods into single versions that auto-inject user_id.
 - Added explanatory comments where user_id is injected and config auto-built.
"""

from pathlib import Path
from platform import processor
from typing import Any, Callable, Dict, List, Optional, Union
# from joblib import Memory
from omegaconf import DictConfig 
from memora.builder.document_memory_builder import DocumentMemoryBuilder
from memora.builder.memory_builder import MemoryBuilder
from memora.builder.memory_builder_registry import MemoryBuilderRegistry
from memora.core.memory import AgentMemory, QueryMode
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig

from memora.builder.chat_memory_builder import ChatMemoryBuilder
from memora.core.memory_entry import MemoryEntry
from memora.core.segment import Segment
from memora.processors.base_processor import detect_file_type, is_supported_file_type
from memora.processors.processor_registry import ProcessorRegistry
from memora.utils.llm import ChatCompletionModel
from memora.utils.log import log_segments
from memora.utils.misc import merge_metadata

class LocalMemoraClient:
    """Client facade exposing memory operations with API key auth only.

    rationale:
    - Users supply only an API key; config is auto-built (no backend leakage).
    - user_id derived from API key is transparently injected into metadata & filters.
    """

    def __init__(
        self,
        cfg: DictConfig,
        user_id: str,
    ):
        """
        Initialize the memory facade.

        Args:
            cfg: Configuration object
            api_key: API key for authentication (will derive user_id)
        """

        # initialize the config
        self.cfg = cfg

        # store the user_id
        self.user_id = user_id

        # initialize the agent memory
        self._agent_memory = AgentMemory(cfg, user_id=user_id)

        # iniialize the model client
        self._model_client = ChatCompletionModel(cfg)

        # memory builder
        self.memory_builder_registry = MemoryBuilderRegistry()
        self.memory_builder_registry.register("markdown", DocumentMemoryBuilder)
        self.memory_builder_registry.register("doc", DocumentMemoryBuilder)
        self.memory_builder_registry.register("chat", ChatMemoryBuilder)
        self.memory_builder_registry.register("default", ChatMemoryBuilder)

        # initialize the memory builder
        # self._memory_builder = ChatMemoryBuilder(cfg, self._agent_memory)

        # initialize the processor registry
        self.processor_registry = ProcessorRegistry()

    def _get_memory_builder(self, file_type: str) -> MemoryBuilder:
        """
        Get the appropriate memory builder for a given file type.

        Args:
            file_type: The detected file type (e.g., 'markdown', 'word', 'pdf')

        Returns:
            MemoryBuilder instance
        """
        # Map file types to memory builder types
        # For now, most file types use the chat builder, but this can be extended
        builder_type_mapping = {
            "default": "chat",
            "chat": "chat",
            "markdown": "markdown",
            "word": "doc",
            "excel": "table",
            "powerpoint": "ppt",
            "pdf": "doc",
            "text": "doc",
            "html": "html",
            "json": "json",
            "yaml": "yaml",
            "xml": "xml",
        }
        print(f"Detected file type: {file_type}")
        builder_type = builder_type_mapping.get(file_type, "default")
        return self.memory_builder_registry.get(
            builder_type, self.cfg, self._agent_memory, self._model_client
        )

    def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> List[MemoryEntry]:
        """Add memory content from a file; automatically stamps user_id.

        Removed duplicate variant: user_id is always derived from API key.
        Insert or update a memory record identified by 'key'.

        Args:
            file_path: Path to the file to add.
            metadata: Additional metadata to store with the memory record

        Returns:
            Record ID (derived from key)
        """
        # Validate file before processing
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not is_supported_file_type(file_path):
            detected_type = detect_file_type(file_path)
            raise ValueError(
                f"Unsupported file type: {detected_type} (file: {file_path.name})"
            )

        segments = self._process_file(file_path)
        log_segments(segments)

        # Detect file type for memory builder selection
        file_type = detect_file_type(file_path)
        memory_builder = self._get_memory_builder(file_type)

        memory_entries = []
        for segment in segments:
            merged_metadata = merge_metadata(segment.metadata, metadata)
            results = memory_builder.build(segment.content, metadata=merged_metadata)
            memory_entries.extend(results)
        return memory_entries

    def _process_file(self, file_path: Union[str, Path]) -> List[Segment]:
        """
        Process a file into segments based on its file type using the processor registry.

        Args:
            file_path: Path to the file to process

        Returns:
            List of Segment objects containing the processed content
        """
        file_path = Path(file_path)

        # get the appropriate processor for the file
        processor = self.processor_registry.get_processor(file_path)

        # process the file using the selected processor
        segments = processor.process(file_path)
        return segments

    def add(
        self,
        text: Union[str, List[str], List[Dict[str, str]]] = None,
        type: str = None,
        metadata: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,  # for progress bar
    ) -> List[MemoryEntry]:
        """Add memory content; automatically stamps user_id.

        Removed duplicate variant: user_id is always derived from API key.
        Insert or update a memory record identified by 'key'.

        Args:
            text: Text to add. Can be:
                - str: Natural language text
                - List[str]: Multiple text entries
                - List[Dict[str, str]]: Structured context with key-value pairs
            metadata: Additional metadata to store with the memory record

        Returns:
            Record ID (derived from key)
        """
        if text is None:
            raise ValueError("Text must be provided")
        
        # TODO: implement the splitting logic for long texts. For now, we treat each text as a single segment.
        segments = [Segment(content=text, segment_type="text", metadata=metadata)]
        
        # get builder based on type        
        memory_builder: MemoryBuilder = self._get_memory_builder(type)
        self._last_builder = memory_builder

        memory_entries = []
        for segment in segments:
            merged_metadata = merge_metadata(segment.metadata, metadata)
            memory_entries.extend(
                memory_builder.build(segment.content, metadata=merged_metadata, progress_callback=progress_callback)
            )

        return memory_entries

    def get_last_build_stats(self):
        """Return build stats from the most recent add() call."""
        if hasattr(self, '_last_builder') and self._last_builder:
            return dict(self._last_builder._build_stats)
        return None

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        query_mode: Optional[QueryMode] = None,
        **kwargs,
    ):
        """
        Vector search by context information to find similar memories.

        Args:
            context: Context to search for. Can be:
                - str: Natural language query text
                - List[str]: Multiple query strings
                - List[Dict[str, str]]: Structured context with key-value pairs
            k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results (e.g., ["metadatas", "distances"])
            enable_hybrid_search: Whether to enable hybrid search combining semantic + keyword search
            enable_llm_filter: Whether to use LLM to filter irrelevant memories
            query_mode: Query mode (ORIGINAL, PRIMARY_ONLY, CUE_ONLY, or BOTH)
            filtering: Whether to apply additional filtering on the retrieved memories

        Returns:
            Backend-specific result object containing matching memories
        """
        # Use config default if not specified
        if query_mode is None:
            query_mode = (
                QueryMode.BOTH
                if self.cfg.memory.enable_cue_index
                else QueryMode.PRIMARY_ONLY
            )
        
        return self._agent_memory.query(
            context,
            top_k=top_k,
            where=where,
            query_mode=query_mode,
            include=include,
            enhance_query=self.cfg.memory.enhance_query,
            return_history=self.cfg.memory.return_history,
            enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter,
            **kwargs,  # Pass through additional parameters like latency_tracker
        )

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """
        List all memory records for the user.

        Args:
            limit: Maximum number of records to return
        Returns:
            List of memory records as dictionaries
        """
        return self._agent_memory.list_memories(limit=limit)

    def get_user_id(self) -> str:
        return self.user_id

    def get(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single record by its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        return self._agent_memory.get(key)

    def delete(self, key: str) -> None:
        """
        Delete a record by its natural-language key.

        Args:
            key: Natural language key to delete
        """
        self._agent_memory.delete(key)

    def count(self) -> int:
        """
        Get the total number of memory records stored.

        Returns:
            Total count of memory records
        """
        return self._agent_memory.count()

    def clear(self) -> None:
        """
        Clear all records in the collection.
        """
        self._agent_memory.clear()

    def delete_all(self, **kwargs) -> None:
        """
        Delete all records for param in the collection.
        """

        if kwargs is None:
            param = {}
        else:
            param = {k: v for k, v in kwargs.items() if v is not None}

        self._agent_memory.delete_all(param)
