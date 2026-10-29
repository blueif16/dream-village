# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
PowerPoint document processor for .ppt and .pptx files.
"""

from pathlib import Path
from typing import List
from memora.core.segment import Segment
from memora.processors.base_processor import FileProcessor, detect_file_type


class PowerPointProcessor(FileProcessor):
    """Processor for Microsoft PowerPoint files (.ppt, .pptx)."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle PowerPoint files."""
        return detect_file_type(file_path) == "powerpoint"

    def process(self, file_path: Path) -> List[Segment]:
        """
        Process PowerPoint file into segments.

        Args:
            file_path: Path to the PowerPoint file

        Returns:
            List of Segment objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError(
                "python-pptx is required to process PowerPoint files. Install with: pip install python-pptx"
            )

        try:
            prs = Presentation(file_path)
            segments = []
            base_metadata = self._create_base_metadata(file_path, "powerpoint")

            for slide_num, slide in enumerate(prs.slides, 1):
                # Slide title
                slide_title = ""
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        if slide_title == "":  # First text is likely the title
                            slide_title = shape.text.strip()
                            segments.append(
                                Segment(
                                    content=slide_title,
                                    segment_type="slide_title",
                                    metadata={
                                        **base_metadata,
                                        "slide_number": slide_num,
                                        "shape_type": str(shape.shape_type),
                                    },
                                )
                            )
                        else:
                            # Other text content
                            segments.append(
                                Segment(
                                    content=shape.text.strip(),
                                    segment_type="slide_content",
                                    metadata={
                                        **base_metadata,
                                        "slide_number": slide_num,
                                        "shape_type": str(shape.shape_type),
                                    },
                                )
                            )

                # If no title found, create a generic slide marker
                if not slide_title:
                    segments.append(
                        Segment(
                            content=f"Slide {slide_num}",
                            segment_type="slide_title",
                            metadata={**base_metadata, "slide_number": slide_num},
                        )
                    )

            return segments

        except Exception as e:
            raise ValueError(f"Failed to process PowerPoint file: {e}")
