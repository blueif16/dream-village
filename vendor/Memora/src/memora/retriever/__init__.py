# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory retrieval strategies."""

from memora.retriever.base_retriever import BaseMemoryRetriever
from memora.retriever.semantic_retriever import SemanticRetriever
from memora.retriever.prompted_policy_retriever import PromptedPolicyRetriever

try:
    from memora.retriever.local_policy_retriever import LocalPolicyRetriever
except ImportError:
    LocalPolicyRetriever = None

__all__ = [
    "BaseMemoryRetriever",
    "SemanticRetriever",
    "PromptedPolicyRetriever",
    "LocalPolicyRetriever",
]
