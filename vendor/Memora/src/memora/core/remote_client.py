# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import requests
import json
from typing import Any, Dict, List, Optional, Union
        
class RemoteMemoraClient:
    """Client facade exposing memory operations with API key auth only.

    rationale:
    - Users supply only an API key; config is auto-built (no backend leakage).
    - user_id derived from API key is transparently injected into metadata & filters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the memory facade.

        Args:
            cfg: Configuration object
            api_key: API key for authentication (will derive user_id)
        """

        # initialize the api_key
        self.api_key = api_key

        # remote server url
        self.server_url = os.getenv("MEMORA_REMOTE_SERVER_URL", "http://localhost:8000")

    def add(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add memory content; automatically stamps user_id.

        Removed duplicate variant: user_id is always derived from API key.
        Insert or update a memory record identified by 'key'.

        Args:
            context: Context to add. Can be:
                - str: Natural language text
                - List[str]: Multiple text entries
                - List[Dict[str, str]]: Structured context with key-value pairs
            metadata: Additional metadata to store with the memory record

        Returns:
            Record ID (derived from key)
        """
        # Initialize metadata if None and add user_id if provided

        
        if metadata is None:
            metadata = {}
        
        # Prepare the request payload
        payload = {
            "context": context,
            "metadata": metadata
        }
        
        # Set up headers with API key authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }
        
        try:
            # Send POST request to the server
            response = requests.post(
                f"{self.server_url}/api/v1/memory/add",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            # Return the response data
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to add memory: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid response format: {str(e)}")

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
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
            filtering: Whether to apply additional filtering on the retrieved memories

        Returns:
            Backend-specific result object containing matching memories
        """

        # Prepare the request payload
        payload = {
            "context": context,
            "top_k": top_k,
            "where": where,
            "include": include,
            "enable_hybrid_search": enable_hybrid_search,
            **kwargs
        }
        
        # Set up headers with API key authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }
        
        try:
            # Send POST request to the server
            response = requests.post(
                f"{self.server_url}/api/v1/memory/query",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            # Return the response data
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to query memories: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid response format: {str(e)}")

    # count, clear, delete_all unchanged (optional: restrict clear/delete_all to admin roles later)

    def get(
        self,
        key: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single record by its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """
        Delete a record by its natural-language key.

        Args:
            key: Natural language key to delete
        """
        raise NotImplementedError

    def count(self) -> int:
        """
        Get the total number of memory records stored.

        Returns:
            Total count of memory records
        """
        raise NotImplementedError

    def clear(self) -> None:
        """
        Clear all records in the collection.
        """
        raise NotImplementedError

    def delete_all(self, **kwargs) -> None:
        """
        Delete all records for param in the collection.
        """

        raise NotImplementedError
