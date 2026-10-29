# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
HTTP-based memory store implementation.
"""

import json
import requests
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

from omegaconf import DictConfig
from chromadb.api.types import Where

from memora.core.base import MemoryBase
from memora.utils.misc import index_to_id

# The system uses LocalMemoryStore directly. Re-enable when externalization is needed.

class HttpMemoryStore(MemoryBase):
    """
    Memory store implementation using HTTP requests to a remote memory service.
    
    This class provides access to a remote memory service via HTTP API calls.
    All memory operations are performed by making requests to a remote server.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the HTTP memory store.
        
        Args:
            cfg: Configuration object containing HTTP memory settings
        """
        self.cfg = cfg

        # Extract HTTP configuration
        self.base_url = cfg.memory.http.base_url.rstrip('/')
        self.timeout = cfg.memory.get('http', {}).get('timeout', 30)
        self.headers = {
            'Content-Type': 'application/json',
        }

        # Add authentication if provided
        if hasattr(cfg.memory.http, 'api_key'):
            self.headers['Authorization'] = f"Bearer {cfg.memory.http.api_key}"
        elif hasattr(cfg.memory.http, 'auth_header'):
            auth_config = cfg.memory.http.auth_header
            self.headers[auth_config.name] = auth_config.value

        # Store collection name for API calls
        self.collection_name = cfg.memory.collection_name

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the memory service.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request payload
            
        Returns:
            Response data as dict
            
        Raises:
            Exception: If request fails
        """
        url = urljoin(self.base_url, endpoint)

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=data, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=self.timeout)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data, timeout=self.timeout)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            # Handle empty responses
            if response.status_code == 204 or not response.content:
                return {}

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse response JSON: {str(e)}")

    def upsert(
        self,
        key: str,
        value: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert or update a memory record identified by 'key'.
        
        Args:
            key: Natural language key to be embedded
            value: Value/content to store
            extra_meta: Additional metadata to store
            
        Returns:
            Record ID (derived from key)
        """
        rid = index_to_id(key)

        # Prepare metadata
        meta = {"original_key": key, "value": value}
        if extra_meta:
            meta = {**meta, **extra_meta}

        # Prepare request data
        data = {
            "collection_name": self.collection_name,
            "id": rid,
            "key": key,
            "value": value,
            "metadata": meta
        }

        # Make upsert request
        endpoint = "/api/memory/upsert"
        self._make_request("POST", endpoint, data)

        return rid

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        k: int = 5,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
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
            
        Returns:
            Query result dict matching ChromaDB format
        """
        include = include or ["metadatas", "distances"]

        # Handle different context types
        if isinstance(context, str):
            query_text = context
        elif isinstance(context, list):
            if all(isinstance(item, str) for item in context):
                # List of strings - join them
                query_text = " ".join(context)
            elif all(isinstance(item, dict) for item in context):
                # List of dictionaries - extract values
                query_text = " ".join([
                    " ".join(item.values()) for item in context
                ])
            else:
                raise ValueError("Context list must contain either all strings or all dictionaries")
        else:
            raise ValueError("Context must be a string, list of strings, or list of dictionaries")

        # Prepare request data
        data = {
            "collection_name": self.collection_name,
            "query_text": query_text,
            "n_results": k,
            "include": include
        }

        if where:
            data["where"] = where

        # Make query request
        endpoint = "/api/memory/query"
        result = self._make_request("POST", endpoint, data)

        return result

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single record by its natural-language key.
        
        Args:
            key: Natural language key to retrieve
            
        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        record_id = index_to_id(key)

        # Prepare request data
        data = {
            "collection_name": self.collection_name,
            "id": record_id
        }

        # Make get request
        endpoint = "/api/memory/get"
        try:
            result = self._make_request("GET", endpoint, data)

            # Handle not found
            if not result or not result.get("found", True):
                return None

            return {
                "id": result.get("id"),
                "metadata": result.get("metadata"),
                "document": result.get("document"),
            }

        except Exception as e:
            # Handle 404 or not found as None
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise

    def delete(self, key: str) -> None:
        """
        Delete a record by its natural-language key.
        
        Args:
            key: Natural language key to delete
        """
        record_id = index_to_id(key)

        # Prepare request data
        data = {
            "collection_name": self.collection_name,
            "id": record_id
        }

        # Make delete request
        endpoint = "/api/memory/delete"
        self._make_request("DELETE", endpoint, data)

    def list_memories(self, limit: int = 10) -> Dict[str, Any]:
        """
        List memories in the collection.

        Args:
            limit: Max number of records to return

        Returns:
            Dict containing memory records
        """
        # Prepare request data
        data = {
            "collection_name": self.collection_name,
            "limit": limit,
            "offset": 0,
            "include": ["documents", "metadatas"]
        }

        # Make list request
        endpoint = "/api/memory/list"
        result = self._make_request("GET", endpoint, data)

        return result

    def count(self) -> int:
        """
        Get the number of records in the collection.

        Returns:
            Number of records in the collection
        """
        # Prepare request data
        data = {
            "collection_name": self.collection_name
        }

        # Make count request
        endpoint = "/api/memory/count"
        result = self._make_request("GET", endpoint, data)

        return result.get("count", 0)
