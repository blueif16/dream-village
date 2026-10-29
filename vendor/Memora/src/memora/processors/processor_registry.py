# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Registry for file processors.
"""

from pathlib import Path
from typing import List, Dict, Type, Optional
from memora.core.segment import Segment
from memora.processors.base_processor import BaseProcessor
from memora.processors.excel_processor import ExcelProcessor
from memora.processors.markdown_processor import MarkdownProcessor
from memora.processors.pdf_processor import PDFProcessor
from memora.processors.powerpoint_processor import PowerPointProcessor
from memora.processors.text_processor import TextProcessor
from memora.processors.word_processor import WordProcessor


class ProcessorRegistry:
    """Registry for managing file processors."""

    def __init__(self):
        """Initialize the processor registry."""
        self._processors: List[BaseProcessor] = []
        self._register_default_processors()

    def _register_default_processors(self):
        """Register default processors."""
        self.register(MarkdownProcessor())
        self.register(TextProcessor())
        self.register(WordProcessor())
        self.register(ExcelProcessor())
        self.register(PowerPointProcessor())
        self.register(PDFProcessor())

    def register(self, processor: BaseProcessor):
        """
        Register a new processor.
        
        Args:
            processor: The processor to register
        """
        self._processors.append(processor)

    def get_processor(self, file_path: Path) -> Optional[BaseProcessor]:
        """
        Get a processor that can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            A processor that can handle the file, or None if no processor found
        """
        for processor in self._processors:
            if processor.can_process(file_path):
                return processor
        return None

    def process_file(self, file_path: Path) -> List[Segment]:
        """
        Process a file using the appropriate processor.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of Segment objects
            
        Raises:
            ValueError: If no processor can handle the file type
        """
        processor = self.get_processor(file_path)
        if processor is None:
            raise ValueError(f"No processor found for file type: {file_path.suffix}")

        return processor.process(file_path)

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        extensions = []
        # This is a simple approach - in practice you might want processors
        # to expose their supported extensions more explicitly
        test_files = [
            Path("test.md"), Path("test.markdown"),
            Path("test.txt"), Path("test.docx"),
            Path("test.pdf"), Path("test.xlsx")
        ]

        for test_file in test_files:
            if self.get_processor(test_file) is not None:
                extensions.append(test_file.suffix)

        return list(set(extensions))


# Global processor registry instance
processor_registry = ProcessorRegistry()
