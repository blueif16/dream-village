# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
PDF document processor for .pdf files.
"""

from pathlib import Path
from typing import List
from memora.core.segment import Segment
from memora.processors.base_processor import FileProcessor, detect_file_type


class PDFProcessor(FileProcessor):
    """Processor for PDF files (.pdf)."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle PDF files."""
        return detect_file_type(file_path) == "pdf"

    def process(self, file_path: Path) -> List[Segment]:
        """
        Process PDF file into segments.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Segment objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required to process PDF files. Install with: pip install pdfplumber"
            )

        try:
            segments = []
            base_metadata = self._create_base_metadata(file_path, "pdf")

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text and text.strip():
                        # Split page content into paragraphs
                        paragraphs = text.split('\n\n')

                        for para_num, paragraph in enumerate(paragraphs, 1):
                            if paragraph.strip():
                                segments.append(
                                    Segment(
                                        content=paragraph.strip(),
                                        segment_type="paragraph",
                                        metadata={
                                            **base_metadata,
                                            "page_number": page_num,
                                            "paragraph_number": para_num,
                                        },
                                    )
                                )

                    # Extract tables if any
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables, 1):
                        if table:
                            # Convert table to text representation
                            table_text = "\n".join(
                                [
                                    "\t".join(
                                        [str(cell) if cell else "" for cell in row]
                                    )
                                    for row in table
                                ]
                            )
                            segments.append(
                                Segment(
                                    content=table_text,
                                    segment_type="table",
                                    metadata={
                                        **base_metadata,
                                        "page_number": page_num,
                                        "table_number": table_num,
                                        "row_count": len(table),
                                        "column_count": len(table[0]) if table else 0,
                                    },
                                )
                            )

            return segments

        except Exception as e:
            raise ValueError(f"Failed to process PDF file: {e}")
