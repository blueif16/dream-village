# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Markdown file processor for converting markdown files to segments.
"""

import re
from pathlib import Path
from typing import List
from markdownify import markdownify
from memora.core.segment import Segment
from memora.processors.base_processor import FileProcessor, detect_file_type


class MarkdownProcessor(FileProcessor):
    """Processor for markdown files (.md, .markdown, .mdown, .mdx)."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle markdown files."""
        return detect_file_type(file_path) == "markdown"

    def process(self, file_path: Path) -> List[Segment]:
        """
        Process markdown content into segments.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            List of Segment objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file_content(file_path)
        return self._process_markdown_content(content, file_path)

    def _process_markdown_content(self, content: str, file_path: Path) -> List[Segment]:
        """
        Process markdown content into segments based on structural boundaries.
        Each segment contains complete content between headings or major structural elements.

        Args:
            content: The markdown content
            file_path: Path to the source file

        Returns:
            List of Segment objects
        """
        # First, generate segments based on original markdown structure (preserves headings)
        segments = []
        lines = content.split("\n")
        base_metadata = self._create_base_metadata(file_path, "markdown")

        current_segment_lines = []
        current_heading = None
        current_heading_level = None

        # Track heading hierarchy - store headings by level
        heading_hierarchy = {}  # {level: heading_text}

        for i, line in enumerate(lines):
            # Check for headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line.strip())

            if heading_match:
                # Save previous segment if it has content
                if current_segment_lines and any(
                    l.strip() for l in current_segment_lines
                ):
                    segment_content = "\n".join(current_segment_lines).strip()
                    segment_metadata = base_metadata.copy()

                    # Always add heading metadata (with dummy values if no heading)
                    segment_metadata = self._add_heading_metadata(
                        segment_metadata,
                        current_heading,
                        current_heading_level,
                        heading_hierarchy,
                    )

                    # Clean HTML tags from the segment content after structure is preserved
                    cleaned_segment_content = self._clean_html_tags(segment_content)

                    segments.append(
                        Segment(
                            content=cleaned_segment_content,
                            segment_type="section",
                            metadata=segment_metadata,
                        )
                    )

                # Start new segment with this heading
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # Update heading hierarchy
                heading_hierarchy[level] = heading_text
                # Remove any deeper level headings
                keys_to_remove = [k for k in heading_hierarchy.keys() if k > level]
                for k in keys_to_remove:
                    del heading_hierarchy[k]

                current_heading = heading_text
                current_heading_level = level
                current_segment_lines = [line]  # Include the heading in the segment

            else:
                # Add line to current segment
                current_segment_lines.append(line)

        # Save final segment if it has content
        if current_segment_lines and any(l.strip() for l in current_segment_lines):
            segment_content = "\n".join(current_segment_lines).strip()
            segment_metadata = base_metadata.copy()

            # Always add heading metadata (with dummy values if no heading)
            segment_metadata = self._add_heading_metadata(
                segment_metadata,
                current_heading,
                current_heading_level,
                heading_hierarchy,
            )

            # Clean HTML tags from the segment content after structure is preserved
            cleaned_segment_content = self._clean_html_tags(segment_content)

            segments.append(
                Segment(
                    content=cleaned_segment_content,
                    segment_type="section",
                    metadata=segment_metadata,
                )
            )

        return segments

    def _clean_html_tags(self, content: str) -> str:
        """
        Remove HTML tags and convert HTML elements to markdown format using markdownify library.

        Args:
            content: The content with potential HTML tags

        Returns:
            Cleaned content with HTML converted to markdown
        """
        # Use markdownify to convert HTML to markdown
        cleaned_content = markdownify(
            content,
            heading_style="ATX",  # Use # style headings
            bullets="-",  # Use - for bullet points
            strip=["script", "style"],  # Remove script and style tags completely
        )

        # Fix potential line break issues from markdownify
        # Ensure headings are on their own lines by inserting line breaks where needed
        cleaned_content = re.sub(r"([.!?])(#{1,6})", r"\1\n\n\2", cleaned_content)
        cleaned_content = re.sub(
            r"(\|)(#{1,6})", r"\1\n\n\2", cleaned_content
        )  # After table rows
        cleaned_content = re.sub(
            r"(\*\*)(#{1,6})", r"\1\n\n\2", cleaned_content
        )  # After bold text

        # Clean up extra whitespace
        cleaned_content = re.sub(
            r"\n\s*\n\s*\n+", "\n\n", cleaned_content
        )  # Multiple empty lines to double
        cleaned_content = re.sub(
            r"^\s+|\s+$", "", cleaned_content, flags=re.MULTILINE
        )  # Trim lines

        return cleaned_content.strip()

    def _build_heading_path(self, heading_hierarchy: dict, current_level: int) -> str:
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

    def _add_heading_metadata(
        self,
        metadata: dict,
        current_heading: str,
        current_heading_level: int,
        heading_hierarchy: dict,
    ) -> dict:
        """
        Add heading metadata to segment metadata, using dummy values if no heading context.

        Args:
            metadata: Base metadata dictionary
            current_heading: Current heading text (None if no heading)
            current_heading_level: Current heading level (None if no heading)
            heading_hierarchy: Dictionary of heading hierarchy

        Returns:
            Updated metadata dictionary with heading fields
        """
        if current_heading:
            # Normal case with heading context
            heading_path = self._build_heading_path(
                heading_hierarchy, current_heading_level
            )
            metadata.update(
                {
                    "heading": current_heading,
                    "heading_level": current_heading_level,
                    "heading_path": heading_path,
                    "parent_headings": heading_hierarchy.copy(),
                }
            )
        else:
            # Dummy values for segments without heading context
            metadata.update(
                {
                    "heading": "",
                    "heading_level": 0,
                    "heading_path": "",
                    "parent_headings": {},
                }
            )

        return metadata
