# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Text file processor for converting plain text files to segments.
"""

from pathlib import Path
from typing import List
from memora.core.segment import Segment
from memora.processors.base_processor import BaseProcessor


class SimpleTextProcessor(BaseProcessor):
    """Processor for plain text files (.txt, .text)."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle text files."""
        return file_path.suffix.lower() in ['.txt', '.text', ''] or file_path.suffix == ''

    def process(self, file_path: Path) -> List[Segment]:
        """
        Process text content into segments.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Segment objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file_content(file_path)
        return self._process_text_content(content, file_path)

    def _process_text_content(self, content: str, file_path: Path) -> List[Segment]:
        """
        Process text content into segments.
        
        Args:
            content: The text content
            file_path: Path to the source file
            
        Returns:
            List of Segment objects
        """
        # Simple paragraph-based segmentation for plain text
        paragraphs = content.split('\n\n')
        segments = []
        base_metadata = self._create_base_metadata(file_path, "text")

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                segments.append(Segment(
                    content=paragraph.strip(),
                    segment_type="paragraph",
                    metadata={
                        **base_metadata,
                        "paragraph_number": i + 1
                    }
                ))

        # If no paragraphs found (single line or no double newlines), treat as single segment
        if not segments and content.strip():
            segments.append(Segment(
                content=content.strip(),
                segment_type="text",
                metadata=base_metadata
            ))

        return segments
