# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import List, Union
from omegaconf import DictConfig
from openai import AzureOpenAI, OpenAI
from azure.identity import (
    get_bearer_token_provider,
    ManagedIdentityCredential,
)


def get_aoai_embedding_client(cfg: DictConfig) -> AzureOpenAI:
    """
    get the azure openai client
    Args:
        cfg: the config

    Returns: the azure openai client
    """
    client_id = cfg.openai.managed_identity
    end_point = cfg.openai.embedding_api_base
    _default_azure_ad_token_provider = get_bearer_token_provider(
        ManagedIdentityCredential(client_id=client_id),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        # model=cfg.embedding.model,
        azure_deployment=cfg.openai.embedding_deployment_name,
        api_version = cfg.openai.embedding_api_version,
        azure_endpoint = end_point,
        azure_ad_token_provider=_default_azure_ad_token_provider,
        
    )
    return client


def get_openai_embedding_client(cfg: DictConfig) -> OpenAI:
    """
    Get the standard OpenAI embedding client (non-Azure).
    Args:
        cfg: the config (requires cfg.openai.api_key or OPENAI_API_KEY env var)

    Returns: the OpenAI client
    """
    api_key = cfg.openai.get("api_key", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key must be provided via cfg.openai.api_key or OPENAI_API_KEY environment variable."
        )
    client = OpenAI(api_key=api_key)
    return client


def get_embedding_client(cfg: DictConfig) -> Union[AzureOpenAI, OpenAI]:
    """
    Get the appropriate embedding client based on api_type config.
    Args:
        cfg: the config

    Returns: AzureOpenAI or OpenAI client
    """
    api_type = cfg.openai.get("api_type", "azure")
    if api_type == "openai":
        return get_openai_embedding_client(cfg)
    else:
        return get_aoai_embedding_client(cfg)


def uses_gemini_embeddings(cfg: DictConfig) -> bool:
    """memo-bench port (port-plan 1b): a Claude backbone embeds with gemini-embedding-2 via memobench.embed."""
    return str(cfg.llm.model).lower().startswith(("claude", "gemini"))


class BaseEmbeddingModel:
    """
    Base class for the language model.
    """

    def __init__(self, cfg: DictConfig, client: Union[AzureOpenAI, OpenAI] = None):
        """
        Args:
            cfg: the config
            client: the OpenAI-compatible client (Azure or standard)
        """
        self.cfg = cfg
        self.use_gemini = uses_gemini_embeddings(cfg)
        if self.use_gemini:
            self.client = None
        else:
            self.client = get_embedding_client(cfg) if not client else client

    def get_client(self) -> Union[AzureOpenAI, OpenAI]:
        """
        Get the OpenAI-compatible client
        Returns: the client
        """
        return self.client

    def generate_embeddings(
        self,
        input: List[str],
        kind: str = "document",
    ) -> List[List[float]]:
        """
        Generate the embeddings for the input text.
        Args:
            input: the input text
            kind: gemini path only: task prefix "document" | "query" | "consolidation" (addendum §5)

        Returns: the response
        """
        if self.use_gemini:
            from memobench.embed import embed

            return embed(list(input), kind)

        model = self.cfg.openai.get("embedding_model", "text-embedding-3-small")
        response = self.client.embeddings.create(input=input, model=model).data

        response = [embedding.embedding for embedding in response]
        return response
