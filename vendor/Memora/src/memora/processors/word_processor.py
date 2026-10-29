# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Word document processor for .doc and .docx files.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from memora.core.segment import Segment
from memora.processors.base_processor import FileProcessor, detect_file_type


class WordProcessor(FileProcessor):
    """Processor for Microsoft Word documents (.doc, .docx)."""

    def __init__(self, max_segment_size: int = 1000):
        """
        Initialize the Word processor.

        Args:
            max_segment_size: Maximum size of each segment in characters (default: 1000)
            combine_siblings: Whether to combine sibling sections (same parent) if total size < max_segment_size (default: False)
        """
        self.max_segment_size = max_segment_size

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle Word files."""
        return detect_file_type(file_path) == "word"

    def process(self, file_path: Path) -> List[Segment]:
        """
        Process Word document into segments, combining headings with their following paragraphs.
        Segments are split when they exceed max_segment_size, respecting paragraph boundaries.

        Args:
            file_path: Path to the Word file

        Returns:
            List of Segment objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required to process Word documents. Install with: pip install python-docx"
            )

        try:
            doc = Document(file_path)
            base_metadata = self._create_base_metadata(file_path, "word")

            # Process paragraphs and combine them into segments
            segments = self._combine_paragraphs_into_segments(doc, base_metadata)

            return segments

        except Exception as e:
            raise ValueError(f"Failed to process Word document: {e}")

    def _combine_paragraphs_into_segments(
        self, doc, base_metadata: Dict[str, Any]
    ) -> List[Segment]:
        """
        Combine headings and paragraphs into segments based on maximum segment size.
        Similar to markdown processor's approach of keeping heading + content together.

        Args:
            doc: Document object from python-docx
            base_metadata: Base metadata for all segments

        Returns:
            List of Segment objects
        """
        segments = []
        current_content_parts = []
        current_heading = None
        current_heading_level = None
        heading_hierarchy = {}  # {level: heading_text}

        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if not text:
                continue

            is_heading = paragraph.style.name.startswith("Heading")

            if is_heading:
                # Extract heading level
                try:
                    level = int(paragraph.style.name.split()[-1])
                except (ValueError, IndexError):
                    level = 1

                # Save previous segment if it has content AND it's not just a heading
                if current_content_parts and len(current_content_parts) > 1:
                    self._save_segment(
                        segments,
                        current_content_parts,
                        current_heading,
                        current_heading_level,
                        heading_hierarchy.copy(),
                        base_metadata,
                    )

                # Update heading hierarchy
                heading_hierarchy[level] = text
                # Remove any deeper level headings
                keys_to_remove = [k for k in heading_hierarchy.keys() if k > level]
                for k in keys_to_remove:
                    del heading_hierarchy[k]

                # Start new segment with this heading
                current_heading = text
                current_heading_level = level
                current_content_parts = [text]

            else:
                # Check if adding this paragraph would exceed max size
                current_size = sum(len(part) for part in current_content_parts)
                if (
                    current_content_parts
                    and len(current_content_parts) > 1
                    and current_size + len(text) + 1 > self.max_segment_size
                ):
                    # Save current segment and start new one
                    self._save_segment(
                        segments,
                        current_content_parts,
                        current_heading,
                        current_heading_level,
                        heading_hierarchy.copy(),
                        base_metadata,
                    )
                    # Start new segment with just this paragraph
                    # Keep heading context but mark as continuation
                    current_content_parts = [text]
                else:
                    # Add paragraph to current segment
                    current_content_parts.append(text)

        # Save final segment if it has content AND it's not just a heading
        if current_content_parts and len(current_content_parts) > 1:
            self._save_segment(
                segments,
                current_content_parts,
                current_heading,
                current_heading_level,
                heading_hierarchy.copy(),
                base_metadata,
            )

        return segments

    def _save_segment(
        self,
        segments: List[Segment],
        content_parts: List[str],
        heading: Optional[str],
        heading_level: Optional[int],
        heading_hierarchy: Dict[int, str],
        base_metadata: Dict[str, Any],
    ) -> None:
        """
        Save a segment with combined content and metadata.

        Args:
            segments: List to append the segment to
            content_parts: List of text parts to combine
            heading: Current heading text
            heading_level: Current heading level
            heading_hierarchy: Dictionary of heading hierarchy
            base_metadata: Base metadata for the segment
        """
        if not content_parts:
            return

        # Combine content parts with newlines
        segment_content = "\n\n".join(content_parts).strip()

        # Build metadata
        metadata = base_metadata.copy()
        metadata.update(self._build_heading_metadata(heading, heading_level, heading_hierarchy))

        # Create segment
        segments.append(
            Segment(
                content=segment_content,
                segment_type="section",
                metadata=metadata,
            )
        )

    def _build_heading_metadata(
        self,
        heading: Optional[str],
        heading_level: Optional[int],
        heading_hierarchy: Dict[int, str],
    ) -> Dict[str, Any]:
        """
        Build heading metadata for a segment.

        Args:
            heading: Current heading text (None if no heading)
            heading_level: Current heading level (None if no heading)
            heading_hierarchy: Dictionary of heading hierarchy

        Returns:
            Dictionary with heading metadata fields
        """
        if heading:
            # Build hierarchical heading path
            heading_path = self._build_heading_path(heading_hierarchy, heading_level)
            return {
                "heading": heading,
                "heading_level": heading_level,
                "heading_path": heading_path,
                "parent_headings": heading_hierarchy.copy(),
            }
        else:
            # Dummy values for segments without heading context
            return {
                "heading": "",
                "heading_level": 0,
                "heading_path": "",
                "parent_headings": {},
            }

    def _build_heading_path(
        self, heading_hierarchy: Dict[int, str], current_level: int
    ) -> str:
        """
        Build a hierarchical heading path from all parent headings.

        Args:
            heading_hierarchy: Dictionary mapping heading levels to heading text
            current_level: The level of the current heading

        Returns:
            String with hierarchical path like "Chapter 1 > Section A > Subsection 3"
        """
        path_parts = []

        # Collect all headings from level 1 up to current level
        for level in sorted(heading_hierarchy.keys()):
            if level <= current_level:
                path_parts.append(heading_hierarchy[level])

        return " > ".join(path_parts)
