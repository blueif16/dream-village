# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base file processor interface for converting files to segments.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from memora.core.segment import Segment


# Single source of truth for file type mappings
_FILE_TYPE_EXTENSIONS = {
    "markdown": [".md", ".markdown", ".mdown", ".mdx"],
    "word": [".doc", ".docx", ".docm", ".dotx", ".dotm"],
    "excel": [".xls", ".xlsx", ".xlsm", ".xlsb", ".xltx", ".xltm", ".csv"],
    "powerpoint": [".ppt", ".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"],
    "pdf": [".pdf"],
    "text": [".txt", ".text", ".log", ".readme"],
    "richtext": [".rtf"],
    "html": [".html", ".htm", ".xhtml"],
    "xml": [".xml"],
    "json": [".json", ".jsonl"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "config": [".ini", ".cfg", ".conf"],
    "code": [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".rs",
    ],
}

# Build reverse mapping once for efficient lookups
_EXTENSION_TO_TYPE = {}
for file_type, extensions in _FILE_TYPE_EXTENSIONS.items():
    for ext in extensions:
        _EXTENSION_TO_TYPE[ext] = file_type


def detect_file_type(file_path: Path) -> str:
    """
    Detect the file type based on the file extension.

    Args:
        file_path: Path to the file

    Returns:
        String representing the file type category, or 'unknown' if not supported
    """
    extension = file_path.suffix.lower()
    return _EXTENSION_TO_TYPE.get(extension, "unknown")


def is_supported_file_type(file_path: Path) -> bool:
    """
    Check if a file type is supported for processing.

    Args:
        file_path: Path to the file

    Returns:
        True if the file type is supported
    """
    extension = file_path.suffix.lower()
    return extension in _EXTENSION_TO_TYPE


def get_supported_extensions() -> Dict[str, List[str]]:
    """
    Get a dictionary of supported file types and their extensions.

    Returns:
        Dictionary mapping file type categories to lists of extensions
    """
    return _FILE_TYPE_EXTENSIONS.copy()  # Return a copy to prevent modification


class BaseProcessor(ABC):
    """Base class for file processors that convert files to segments."""

    ...


class FileProcessor(BaseProcessor):
    """Base class for file processors that convert files to segments."""

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this processor can handle the file
        """
        pass

    @abstractmethod
    def process(self, file_path: Path) -> List[Segment]:
        """
        Process a file into segments.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of Segment objects
        """
        pass

    def _read_file_content(self, file_path: Path) -> str:
        """
        Helper method to read file content with encoding fallback.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    def _create_base_metadata(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """
        Create base metadata for segments.
        
        Args:
            file_path: Path to the source file
            file_type: Type of the file
            
        Returns:
            Dictionary with base metadata
        """
        return {
            "source_file": str(file_path),
            "file_type": file_type,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0
        }
