# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Factory for creating vector database clients based on configuration.
"""
from omegaconf import DictConfig

from memora.db_clients.base import VectorDBClient
from memora.db_clients.chromadb_client import ChromaDBClient
from memora.db_clients.redis_client import RedisVectorDBClient


def create_vector_db_client(cfg: DictConfig) -> VectorDBClient:
    """
    Factory function to create a vector database client based on configuration.
    
    Args:
        cfg: Configuration object containing database settings
        
    Returns:
        VectorDBClient instance (ChromaDB or Redis)
        
    Raises:
        ValueError: If unsupported database type is specified
    """
    db_type = cfg.memory.get("db_type", "chromadb").lower()
    
    if db_type == "chromadb" or db_type == "chroma":
        return ChromaDBClient(cfg)
    elif db_type == "redis":
        return RedisVectorDBClient(cfg)
    else:
        raise ValueError(
            f"Unsupported database type: {db_type}. "
            f"Supported types: 'chromadb', 'redis'"
        )
