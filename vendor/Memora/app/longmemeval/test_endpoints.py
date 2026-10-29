# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python3
"""
Quick script to test Azure OpenAI endpoints with different models.

Usage:
    python test_endpoints.py
    python test_endpoints.py --endpoints "https://ep1.openai.azure.com/" "https://ep2.openai.azure.com/"
"""

import os
import sys

try:
    from openai import AzureOpenAI
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    print("Error: Install required packages: pip install openai azure-identity")
    sys.exit(1)

# Default endpoints to test - override via command line or environment
DEFAULT_ENDPOINTS = [
    os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
]

# Models to test
CHAT_MODELS = ["gpt-4o-mini", "gpt-4.1-mini"]
EMBEDDING_MODELS = ["text-embedding-3-small"]

API_VERSION = "2024-12-01-preview"
MANAGED_IDENTITY = os.environ.get("AZURE_MANAGED_IDENTITY_CLIENT_ID", "")


def test_chat_model(endpoint: str, model: str) -> tuple:
    """Test a single endpoint-model combination for chat."""
    try:
        credential = DefaultAzureCredential(
            managed_identity_client_id=MANAGED_IDENTITY if MANAGED_IDENTITY else None
        )
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=5,
        )

        result = response.choices[0].message.content
        return True, f"OK - Response: {result}"

    except Exception as e:
        return False, f"FAILED - {str(e)[:100]}"


def test_embedding_model(endpoint: str, model: str) -> tuple:
    """Test a single endpoint-model combination for embeddings."""
    try:
        credential = DefaultAzureCredential(
            managed_identity_client_id=MANAGED_IDENTITY if MANAGED_IDENTITY else None
        )
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION,
        )

        response = client.embeddings.create(
            model=model,
            input="Hello world",
        )

        dim = len(response.data[0].embedding)
        return True, f"OK - Dimension: {dim}"

    except Exception as e:
        return False, f"FAILED - {str(e)[:100]}"


def main():
    # Parse endpoints from command line or use defaults
    if "--endpoints" in sys.argv:
        idx = sys.argv.index("--endpoints")
        endpoints = sys.argv[idx + 1:]
    else:
        endpoints = [ep for ep in DEFAULT_ENDPOINTS if ep]

    if not endpoints:
        print("No endpoints configured.")
        print("Set AZURE_OPENAI_ENDPOINT environment variable or pass --endpoints")
        sys.exit(1)

    print(f"Testing {len(endpoints)} endpoint(s)...")
    print(f"Managed Identity: {MANAGED_IDENTITY[:8]}..." if MANAGED_IDENTITY else "Using default credentials")
    print(f"API Version: {API_VERSION}")
    print("=" * 70)

    results = {}
    for endpoint in endpoints:
        print(f"\nEndpoint: {endpoint}")
        print("-" * 50)

        endpoint_results = {"chat": {}, "embedding": {}}

        for model in CHAT_MODELS:
            success, msg = test_chat_model(endpoint, model)
            status = "✓" if success else "✗"
            print(f"  {status} Chat [{model}]: {msg}")
            endpoint_results["chat"][model] = success

        for model in EMBEDDING_MODELS:
            success, msg = test_embedding_model(endpoint, model)
            status = "✓" if success else "✗"
            print(f"  {status} Embedding [{model}]: {msg}")
            endpoint_results["embedding"][model] = success

        results[endpoint] = endpoint_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for endpoint, result in results.items():
        chat_ok = all(result["chat"].values())
        embed_ok = all(result["embedding"].values())
        status = "✓ READY" if (chat_ok and embed_ok) else "✗ ISSUES"
        print(f"  {status}: {endpoint}")


if __name__ == "__main__":
    main()
