# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Redis implementation of the vector database client using Redis Stack.
Requires redis-py package version 4.0+
Install with: pip install redis
"""
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
import json

# Check if redis is available
REDIS_AVAILABLE = False
try:
    import numpy as np
    from redis import Redis
    from redis.commands.search.field import TextField, VectorField, NumericField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Redis not installed or wrong version
    pass

from memora.db_clients.base import VectorDBClient
from memora.utils.embedding import BaseEmbeddingModel


class RedisVectorDBClient(VectorDBClient):
    """Redis Stack implementation of the vector database client."""
    
    # Default metadata fields to index with TAG fields for efficient filtering
    # These fields will have both TEXT (for search) and TAG (for exact filtering) indexes
    DEFAULT_INDEXED_FIELDS = [
        "user_id",
        "linked_memory",
        "creation_time",
        "cue_indices",
    ]

    def __init__(self, cfg: DictConfig):
        """
        Initialize Redis client for vector operations.

        Args:
            cfg: Configuration object containing database settings
        """
        self.cfg = cfg
        
        # Get Redis connection parameters from config
        redis_host = cfg.memory.get("redis_host", "localhost")
        redis_port = cfg.memory.get("redis_port", 6379)
        redis_db = cfg.memory.get("redis_db", 0)
        redis_password = cfg.memory.get("redis_password", None)
        
        # Initialize Redis client
        self.client = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False  # We'll handle encoding manually
        )
        
        # Initialize embedding model
        self.embedding_model = BaseEmbeddingModel(cfg)
        
        # Get embedding dimensions
        self.embedding_dim = cfg.memory.get("embedding_dim", 1536)
        
        # Distance metric (cosine, l2, ip)
        distance_metric = cfg.memory.get("distance", "cosine").upper()
        if distance_metric == "COSINE":
            self.distance_metric = "COSINE"
        elif distance_metric == "L2":
            self.distance_metric = "L2"
        else:
            self.distance_metric = "IP"  # Inner Product
        
        # Use the default list of metadata fields to index
        self.indexed_fields = self.DEFAULT_INDEXED_FIELDS
        
        # Store active collections
        self._collections = {}

    def _get_collection_prefix(self, collection_name: str) -> str:
        """Get the Redis key prefix for a collection."""
        return f"collection:{collection_name}"

    def _get_index_name(self, collection_name: str) -> str:
        """Get the Redis search index name for a collection."""
        return f"idx:{collection_name}"

    def _get_doc_key(self, collection_name: str, doc_id: str) -> str:
        """Get the full Redis key for a document."""
        return f"{self._get_collection_prefix(collection_name)}:doc:{doc_id}"

    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
        """
        Get or create a Redis collection (search index).

        Args:
            collection_name: Name of the collection
            metadata: Collection metadata (e.g., distance metric)

        Returns:
            Collection info dictionary
        """
        index_name = self._get_index_name(collection_name)
        
        # Check if index already exists
        try:
            info = self.client.ft(index_name).info()
            # Index exists, return collection info
            self._collections[collection_name] = {
                "name": collection_name,
                "index_name": index_name,
                "metadata": metadata
            }
            return self._collections[collection_name]
        except Exception:
            # Index doesn't exist, create it
            pass
        
        # Import TagField for efficient filtering
        from redis.commands.search.field import TagField
        
        # Create base schema fields (always indexed)
        schema_fields = [
            TextField("$.id", as_name="id"),
            TextField("$.document", as_name="document"),
            TextField("$.index", as_name="index"),
            TextField("$.value", as_name="value"),
            VectorField(
                "$.embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": self.distance_metric,
                },
                as_name="embedding"
            ),
            NumericField("$.timestamp", as_name="timestamp"),
        ]
        
        # Add dynamic TAG and TEXT fields for configured metadata fields
        for field_name in self.indexed_fields:
            # Add TextField for search
            schema_fields.append(TextField(f"$.{field_name}", as_name=field_name))
            # Add TagField for exact filtering
            schema_fields.append(TagField(f"$.{field_name}", as_name=f"{field_name}_tag"))
        
        schema = tuple(schema_fields)
        
        # Create index definition
        definition = IndexDefinition(
            prefix=[f"{self._get_collection_prefix(collection_name)}:doc:"],
            index_type=IndexType.JSON
        )
        
        # Create the index
        self.client.ft(index_name).create_index(
            fields=schema,
            definition=definition
        )
        
        # Re-index any existing documents (in case index was recreated)
        prefix = self._get_collection_prefix(collection_name)
        existing_keys = self.client.keys(f"{prefix}:doc:*")
        if existing_keys:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Re-indexing {len(existing_keys)} existing documents for collection {collection_name}")
            for key in existing_keys:
                # Fetch and re-set the document to trigger indexing
                doc = self.client.json().get(key)
                if doc:
                    self.client.json().set(key, "$", doc)
        
        # Store collection info
        self._collections[collection_name] = {
            "name": collection_name,
            "index_name": index_name,
            "metadata": metadata
        }
        
        return self._collections[collection_name]

    def upsert(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """
        Insert or update records in the Redis collection.

        Args:
            collection: Collection info dictionary
            ids: List of record IDs
            documents: List of document texts
            metadatas: List of metadata dicts
            embeddings: Optional pre-computed embeddings
        """
        collection_name = collection["name"]
        
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.embedding_model.generate_embeddings(documents)
        
        # Insert each document
        for doc_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            doc_key = self._get_doc_key(collection_name, doc_id)
            
            # Prepare document data
            # Convert embedding to list for JSON serialization
            embedding_list = np.array(embedding, dtype=np.float32).tolist()
            
            # Ensure timestamp is a number for numeric field indexing
            timestamp_val = metadata.get("timestamp", 0)
            if isinstance(timestamp_val, str):
                try:
                    timestamp_val = float(timestamp_val) if timestamp_val else 0
                except:
                    timestamp_val = 0
            
            # For TAG fields, use special marker for empty strings to enable native filtering
            # TAG fields can't index empty strings, so we use "__EMPTY__" as a placeholder
            def tag_value(val):
                return "__EMPTY__" if val == "" or val is None else val
            
            doc_data = {
                "id": doc_id,
                "document": document,
                "index": metadata.get("index", ""),
                "value": metadata.get("value", ""),
                "embedding": embedding_list,  # Store as list for JSON
                "timestamp": timestamp_val,
            }
            
            # Add all metadata fields with TAG-friendly values
            # Apply tag_value to all string metadata to support TAG field filtering
            for key, val in metadata.items():
                if key not in doc_data and key != "timestamp":  # timestamp already handled
                    # Apply tag_value transformation to string values for TAG compatibility
                    if isinstance(val, str):
                        doc_data[key] = tag_value(val)
                    else:
                        doc_data[key] = val
            
            # Store as JSON
            self.client.json().set(doc_key, "$", doc_data)

    def query(
        self,
        collection,
        query_texts: str,
        n_results: int,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the Redis collection for similar vectors.

        Args:
            collection: Collection info dictionary
            query_texts: Query string
            n_results: Number of results to return
            where: Filter conditions (Redis filter expressions)
            include: Fields to include in results

        Returns:
            Query results dictionary in ChromaDB-compatible format
        """
        index_name = collection["index_name"]
        collection_name = collection["name"]
        
        # Generate query embedding
        query_embedding = self.embedding_model.generate_embeddings([query_texts])[0]
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Build filter string and track post-filters for conditions Redis can't express
        filter_parts = []
        post_filters = []  # Filters to apply after retrieval
        
        def escape_redis_value(val: str) -> str:
            """Escape special characters in Redis TAG field values"""
            # Redis TAG fields require escaping for: , . < > { } [ ] " ' : ; ! @ # $ % ^ & * ( ) - + = ~ |
            # Replace spaces and special chars that might cause syntax errors
            special_chars = [',', '.', '<', '>', '{', '}', '[', ']', '"', "'", ':', ';', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', '=', '~', '|', ' ']
            escaped = val
            for char in special_chars:
                escaped = escaped.replace(char, f'\\{char}')
            return escaped
        
        def process_condition(key, value):
            """Process a single condition and add to filter_parts or post_filters"""
            if isinstance(value, dict):
                # Handle ChromaDB operators like {"$ne": ""}, {"$eq": ""}
                if "$ne" in value:
                    ne_val = value["$ne"]
                    if ne_val == "":
                        # Not equal to empty string = has a value
                        # Use TAG field with wildcard to match non-empty values
                        tag_field = f"{key}_tag"
                        filter_parts.append(f"@{tag_field}:{{*}}")
                    else:
                        # Not equal to a specific value - use TAG field with negation
                        # Redis TAG fields support -@field:{value} for negation
                        tag_field = f"{key}_tag"
                        # Escape special characters in the value
                        escaped_val = escape_redis_value(str(ne_val))
                        filter_parts.append(f"-@{tag_field}:{{{escaped_val}}}")
                elif "$eq" in value:
                    eq_val = value["$eq"]
                    if eq_val == "":
                        # Equal to empty string - use special __EMPTY__ marker in TAG field
                        tag_field = f"{key}_tag"
                        filter_parts.append(f"@{tag_field}:{{__EMPTY__}}")
                    else:
                        # Non-empty string equality - use TAG field for exact match
                        tag_field = f"{key}_tag"
                        escaped_val = escape_redis_value(str(eq_val))
                        filter_parts.append(f"@{tag_field}:{{{escaped_val}}}")
                elif "$gt" in value:
                    filter_parts.append(f"@{key}:[({value['$gt']} +inf]")
                elif "$gte" in value:
                    filter_parts.append(f"@{key}:[{value['$gte']} +inf]")
                elif "$lt" in value:
                    filter_parts.append(f"@{key}:[-inf ({value['$lt']}]")
                elif "$lte" in value:
                    filter_parts.append(f"@{key}:[-inf {value['$lte']}]")
            elif isinstance(value, str):
                # Direct string value - use TAG field for exact match
                tag_field = f"{key}_tag"
                escaped_val = escape_redis_value(value)
                filter_parts.append(f"@{tag_field}:{{{escaped_val}}}")
            elif isinstance(value, (int, float)):
                filter_parts.append(f"@{key}:[{value} {value}]")
            else:
                tag_field = f"{key}_tag"
                filter_parts.append(f"@{tag_field}:{{{str(value)}}}")
        
        if where:
            # Handle ChromaDB logical operators
            if "$and" in where:
                # Process all conditions in the $and list
                for condition in where["$and"]:
                    for key, value in condition.items():
                        process_condition(key, value)
            elif "$or" in where:
                # OR in Redis - build multiple filter parts and combine with pipe (|)
                or_parts = []
                for condition in where["$or"]:
                    for key, value in condition.items():
                        if isinstance(value, dict):
                            if "$eq" in value and value["$eq"] != "":
                                tag_field = f"{key}_tag"
                                escaped_val = escape_redis_value(str(value["$eq"]))
                                or_parts.append(f"@{tag_field}:{{{escaped_val}}}")
                            elif "$ne" in value and value["$ne"] == "":
                                tag_field = f"{key}_tag"
                                or_parts.append(f"@{tag_field}:{{*}}")
                            else:
                                # Complex OR condition - fallback to post-filter
                                for op, op_val in value.items():
                                    post_filters.append((key, op, op_val))
                        elif isinstance(value, str):
                            tag_field = f"{key}_tag"
                            escaped_val = escape_redis_value(value)
                            or_parts.append(f"@{tag_field}:{{{escaped_val}}}")
                # Combine OR parts with pipe
                if or_parts:
                    filter_parts.append("(" + "|".join(or_parts) + ")")
            else:
                # Simple conditions without logical operators
                for key, value in where.items():
                    process_condition(key, value)
        
        # Build KNN query with smart filtering
        # Combine Redis filters with KNN, use post-filters only when necessary
        if filter_parts:
            # We have Redis-native filters - use them!
            filter_str = "(" + " ".join(filter_parts) + ")"
            if post_filters:
                # Have both Redis filters and post-filters
                # Redis filters reduce the search space, then post-filter
                # Use smaller multiplier since Redis already filtered
                fetch_count = n_results * 2
            else:
                # Only Redis filters - no post-filtering needed
                fetch_count = n_results
            query_str = f"{filter_str}=>[KNN {fetch_count} @embedding $vector AS score]"
        else:
            # No Redis filters, only post-filters or no filters
            if post_filters:
                # Only post-filters - need to over-fetch
                fetch_count = n_results * 5  # More conservative multiplier
            else:
                # No filters at all
                fetch_count = n_results
            query_str = f"*=>[KNN {fetch_count} @embedding $vector AS score]"
        
        # Build return fields list dynamically
        return_fields = ["id", "document", "index", "value", "score", "timestamp"] + self.indexed_fields
        
        query = (
            Query(query_str)
            .sort_by("score")
            .return_fields(*return_fields)
            .dialect(2)
        )
        
        # Execute query
        results = self.client.ft(index_name).search(
            query,
            query_params={"vector": query_vector}
        )
        
        # Convert to ChromaDB-compatible format and apply post-filters
        ids = []
        metadatas = []
        distances = []
        documents = []
        
        for doc in results.docs:
            doc_id = doc.id.split(":")[-1]  # Extract doc_id from key
            
            # Fetch the full JSON document to get all fields
            # Redis search results may not include all fields properly
            doc_key = self._get_doc_key(collection_name, doc_id)
            full_doc = self.client.json().get(doc_key)
            
            if not full_doc:
                continue
            
            # Build metadata - include all fields from MemoryEntry
            # Ensure timestamp is a string (MemoryEntry expects string)
            timestamp = full_doc.get("timestamp", 0)
            timestamp_str = str(timestamp) if timestamp else ""
            
            # Convert __EMPTY__ marker back to empty strings
            def from_tag_value(val):
                return "" if val == "__EMPTY__" else (val or "")
            
            metadata = {
                "index": full_doc.get("index", ""),
                "value": full_doc.get("value", ""),
                "timestamp": timestamp_str,
            }
            
            # Add all other metadata fields, converting TAG values back to normal
            for key, value in full_doc.items():
                if key not in ["id", "document", "embedding", "index", "value", "timestamp"]:
                    # Convert __EMPTY__ marker back to empty string for all string fields
                    if isinstance(value, str):
                        metadata[key] = from_tag_value(value)
                    else:
                        metadata[key] = value
            
            # Apply post-filters
            skip = False
            for filter_key, filter_op, filter_val in post_filters:
                field_value = metadata.get(filter_key, "")
                if filter_op == "$ne":
                    if field_value == filter_val:
                        skip = True
                        break
                elif filter_op == "$eq":
                    if field_value != filter_val:
                        skip = True
                        break
            
            if skip:
                continue
            
            ids.append(doc_id)
            metadatas.append(metadata)
            
            # Score to distance conversion
            score = float(getattr(doc, "score", 0))
            if self.distance_metric == "COSINE":
                # Redis returns 1 - cosine_similarity as score
                distance = score
            else:
                distance = score
            distances.append(distance)
            
            documents.append(full_doc.get("document", ""))
            
            # Stop once we have enough results after filtering
            if len(ids) >= n_results:
                break
        
        return {
            "ids": [ids],
            "metadatas": [metadatas],
            "distances": [distances],
            "documents": [documents]
        }

    def get(
        self,
        collection,
        ids: Optional[List[str]] = None,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get records from the Redis collection.

        Args:
            collection: Collection info dictionary
            ids: Optional list of IDs to retrieve
            where: Filter conditions
            include: Fields to include in results
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Records dictionary in ChromaDB-compatible format
        """
        collection_name = collection["name"]
        index_name = collection["index_name"]
        
        # Helper to convert __EMPTY__ marker back to empty strings
        def from_tag_value(val):
            return "" if val == "__EMPTY__" else (val or "")
        
        result_ids = []
        result_metadatas = []
        result_documents = []
        
        if ids:
            # Retrieve specific documents by ID
            for doc_id in ids:
                doc_key = self._get_doc_key(collection_name, doc_id)
                doc_data = self.client.json().get(doc_key)
                
                if doc_data:
                    result_ids.append(doc_id)
                    
                    # Extract metadata - ensure timestamp is string and convert TAG values
                    timestamp = doc_data.get("timestamp", 0)
                    timestamp_str = str(timestamp) if timestamp else ""
                    
                    metadata = {
                        "index": doc_data.get("index", ""),
                        "value": doc_data.get("value", ""),
                        "timestamp": timestamp_str,
                    }
                    # Add any additional metadata fields, converting TAG values
                    for key, value in doc_data.items():
                        if key not in ["id", "document", "embedding", "index", "value", "timestamp"]:
                            # Convert __EMPTY__ marker back to empty string for all string fields
                            if isinstance(value, str):
                                metadata[key] = from_tag_value(value)
                            else:
                                metadata[key] = value
                    
                    result_metadatas.append(metadata)
                    result_documents.append(doc_data.get("document", ""))
        else:
            # Search for all documents or with filters
            filter_str = "*"
            if where:
                filters = []
                for key, value in where.items():
                    if isinstance(value, str):
                        filters.append(f"@{key}:{{{value}}}")
                    else:
                        filters.append(f"@{key}:[{value} {value}]")
                if filters:
                    filter_str = " ".join(filters)
            
            # Build query with dynamic return fields
            return_fields = ["id", "index", "value", "document", "timestamp"] + self.indexed_fields
            query = Query(filter_str).return_fields(*return_fields)
            
            if limit:
                query = query.paging(offset or 0, limit)
            
            # Execute search
            results = self.client.ft(index_name).search(query)
            
            for doc in results.docs:
                doc_id = doc.id.split(":")[-1]
                result_ids.append(doc_id)
                
                # Get full document data
                doc_key = self._get_doc_key(collection_name, doc_id)
                doc_data = self.client.json().get(doc_key)
                
                if doc_data:
                    # Ensure timestamp is string and convert TAG values
                    timestamp = doc_data.get("timestamp", 0)
                    timestamp_str = str(timestamp) if timestamp else ""
                    
                    metadata = {
                        "index": doc_data.get("index", ""),
                        "value": doc_data.get("value", ""),
                        "timestamp": timestamp_str,
                    }
                    # Add any additional metadata fields, converting TAG values
                    for key, value in doc_data.items():
                        if key not in ["id", "document", "embedding", "index", "value", "timestamp"]:
                            # Convert __EMPTY__ marker back to empty string for all string fields
                            if isinstance(value, str):
                                metadata[key] = from_tag_value(value)
                            else:
                                metadata[key] = value
                    
                    result_metadatas.append(metadata)
                    result_documents.append(doc_data.get("document", ""))
        
        return {
            "ids": result_ids,
            "metadatas": result_metadatas,
            "documents": result_documents
        }

    def delete(self, collection, ids: List[str]):
        """
        Delete records from the Redis collection.

        Args:
            collection: Collection info dictionary
            ids: List of record IDs to delete
        """
        collection_name = collection["name"]
        
        for doc_id in ids:
            doc_key = self._get_doc_key(collection_name, doc_id)
            self.client.delete(doc_key)

    def count(self, collection) -> int:
        """
        Get the count of records in the Redis collection.

        Args:
            collection: Collection info dictionary

        Returns:
            Number of records
        """
        index_name = collection["index_name"]
        
        try:
            # Search for all documents
            results = self.client.ft(index_name).search(Query("*").paging(0, 0))
            return results.total
        except Exception:
            return 0

    def delete_collection(self, collection_name: str):
        """
        Delete an entire Redis collection (search index and all documents).

        Args:
            collection_name: Name of the collection to delete
        """
        index_name = self._get_index_name(collection_name)
        prefix = f"{self._get_collection_prefix(collection_name)}:doc:*"
        
        # Drop the index
        try:
            self.client.ft(index_name).dropindex(delete_documents=True)
        except Exception:
            pass
        
        # Delete all documents with this prefix
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=prefix, count=100)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break
        
        # Remove from collections dict
        if collection_name in self._collections:
            del self._collections[collection_name]
