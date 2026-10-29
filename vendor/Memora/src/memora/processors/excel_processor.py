# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Excel document processor for .xls, .xlsx, and .csv files.
"""

from pathlib import Path
from typing import List
from memora.core.segment import Segment
from memora.processors.base_processor import FileProcessor, detect_file_type


class ExcelProcessor(FileProcessor):
    """Processor for Microsoft Excel files (.xls, .xlsx, .csv)."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle Excel files."""
        return detect_file_type(file_path) == "excel"

    def process(self, file_path: Path) -> List[Segment]:
        """
        Process Excel file into segments.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of Segment objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension == ".csv":
            return self._process_csv(file_path)
        else:
            return self._process_excel(file_path)

    def _process_csv(self, file_path: Path) -> List[Segment]:
        """Process CSV file."""
        import csv

        segments = []
        base_metadata = self._create_base_metadata(file_path, "excel")

        with open(file_path, "r", encoding="utf-8", newline="") as csvfile:
            # Detect dialect
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

            reader = csv.reader(csvfile, dialect)
            rows = list(reader)

            if rows:
                # First row as headers
                headers = rows[0]
                segments.append(
                    Segment(
                        content=", ".join(headers),
                        segment_type="table_header",
                        metadata={
                            **base_metadata,
                            "row_number": 1,
                            "column_count": len(headers),
                        },
                    )
                )

                # Data rows
                for i, row in enumerate(rows[1:], 2):
                    if any(cell.strip() for cell in row):  # Skip empty rows
                        segments.append(
                            Segment(
                                content=", ".join(row),
                                segment_type="table_row",
                                metadata={
                                    **base_metadata,
                                    "row_number": i,
                                    "column_count": len(row),
                                },
                            )
                        )

        return segments

    def _process_excel(self, file_path: Path) -> List[Segment]:
        """Process Excel file (.xls, .xlsx)."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is required to process Excel files. Install with: pip install openpyxl"
            )

        segments = []
        base_metadata = self._create_base_metadata(file_path, "excel")

        try:
            workbook = openpyxl.load_workbook(file_path, data_only=True)

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]

                # Sheet as a segment
                segments.append(
                    Segment(
                        content=f"Worksheet: {sheet_name}",
                        segment_type="worksheet",
                        metadata={
                            **base_metadata,
                            "sheet_name": sheet_name,
                            "max_row": sheet.max_row,
                            "max_column": sheet.max_column,
                        },
                    )
                )

                # Process rows with data
                for row_num, row in enumerate(sheet.iter_rows(values_only=True), 1):
                    if any(cell is not None and str(cell).strip() for cell in row):
                        row_content = ", ".join(
                            str(cell) if cell is not None else "" for cell in row
                        )
                        segments.append(
                            Segment(
                                content=row_content,
                                segment_type="table_row",
                                metadata={
                                    **base_metadata,
                                    "sheet_name": sheet_name,
                                    "row_number": row_num,
                                    "column_count": len(row),
                                },
                            )
                        )

            return segments

        except Exception as e:
            raise ValueError(f"Failed to process Excel file: {e}")
