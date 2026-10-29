# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import json
import logging
import os
import string
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
from omegaconf import DictConfig
from openai import AzureOpenAI, OpenAI, ContentFilterFinishReasonError
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider

from memora.utils.token_usage import TokenUsageCallback

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- Anthropic path (memo-bench port)
# Plumbing only: prompt text and call-site behavior are unchanged (docs/port-plan.md Part 1a).

ANTHROPIC_MAX_TOKENS = 8192  # explicit; thinking counts toward it
ANTHROPIC_EFFORT = "low"
_FALLBACK_STOP_REASONS = ("max_tokens", "refusal")

# stage = what the call is for (ledger/log tag). Explicit `source` wins, then the response_format class.
_STAGE_BY_SOURCE = {
    "ConversationSegmenter": "segmentation",
    "PromptedPolicyRetriever": "policy",
}
_STAGE_BY_FORMAT = {
    "SegmentationOutput": "segmentation",
    "MemoryOutputs": "extraction",
    "EpisodicMemoryOutput": "episodic",
    "BatchCueIndices": "cue",
    "MemoryUpdateDecision": "update",
}

_CACHE_PREFIXES: Dict[str, str] = {}  # template name -> static prompt head to mark with cache_control
_dropped_logged: set = set()
_log_lock = threading.Lock()


def register_cache_prefix(name: str, template: str) -> None:
    """Cache the static head of `template` wherever a user prompt starts with it: the literal text before its
    first `{placeholder}`, as str.format renders it (escaped `{{`/`}}` become single braces)."""
    prefix = ""
    for literal, field, _, _ in string.Formatter().parse(template):
        prefix += literal
        if field is not None:
            break
    _CACHE_PREFIXES[name] = prefix


def _stage_for(source: str, response_format: Any) -> str:
    if source in _STAGE_BY_SOURCE:
        return _STAGE_BY_SOURCE[source]
    return _STAGE_BY_FORMAT.get(getattr(response_format, "__name__", ""), "other")


def _log_dropped_once(name: str) -> None:
    with _log_lock:
        if name in _dropped_logged:
            return
        _dropped_logged.add(name)
    logger.warning(f"Anthropic path: dropping OpenAI-only argument '{name}' (logged once)")


def _memobench_root() -> Path:
    from memobench import runtime

    return Path(os.environ.get("MEMOBENCH_ROOT") or (runtime.REPO / "runs"))


def _log_parse_failure(record: Dict[str, Any]) -> None:
    path = _memobench_root() / "logs" / "parse_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _empty_response(response_format: Any, error: Exception) -> Any:
    """What upstream returns on a content-filter error (see _invoke_azure): an empty instance, or re-raise."""
    try:
        return response_format(entries=[])
    except Exception:
        try:
            return response_format()
        except Exception:
            logger.error(f"Could not create empty response for format {response_format}")
            raise error


def _to_anthropic_messages(messages: list) -> tuple:
    """OpenAI-style message list -> (system, messages). Leading system messages become `system`;
    OpenAI `image_url` parts become Anthropic URL image blocks; text is passed through unchanged."""
    system_parts = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        content = messages[i]["content"]
        system_parts.append(content if isinstance(content, str) else "".join(p.get("text", "") for p in content))
        i += 1
    converted = []
    for msg in messages[i:]:
        content = msg["content"]
        if isinstance(content, list):
            blocks = []
            for part in content:
                if part.get("type") == "image_url":
                    blocks.append({"type": "image", "source": {"type": "url", "url": part["image_url"]["url"]}})
                else:
                    blocks.append(part)
            content = blocks
        converted.append({"role": msg["role"], "content": content})
    return ("\n\n".join(system_parts) or None), converted


def _apply_cache_split(messages: list, source: str) -> list:
    """Split the user prompt into [static prefix + cache_control][remainder] when it starts with a registered
    template head (longest match); the concatenation is unchanged."""
    if len(messages) != 1 or not isinstance(messages[0]["content"], str):
        return messages
    text = messages[0]["content"]
    prefix = max((p for p in _CACHE_PREFIXES.values() if p and text.startswith(p)), key=len, default=None)
    if not prefix:
        return messages
    blocks = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}]
    if len(text) > len(prefix):
        blocks.append({"type": "text", "text": text[len(prefix):]})
    return [{"role": messages[0]["role"], "content": blocks}]


def get_aoai_chat_completion_client(cfg: DictConfig) -> AzureOpenAI:
    """
    get the azure openai client
    Args:
        cfg: the config

    Returns: the azure openai client
    """
    _default_azure_ad_token_provider = get_bearer_token_provider(
        ManagedIdentityCredential(client_id=cfg.openai.managed_identity),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        api_version=cfg.openai.llm_api_version,
        azure_endpoint=cfg.openai.llm_api_base,
        azure_ad_token_provider=_default_azure_ad_token_provider,
    )
    
    return client


def get_openai_chat_completion_client(cfg: DictConfig) -> OpenAI:
    """
    Get the standard OpenAI client (non-Azure).
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


class ChatCompletionModel:
    """
    Base class for the language model.
    Supports both Azure OpenAI and local Hugging Face models (e.g., Qwen3-8B).
    Model type is automatically determined from the model name.
    """

    def __init__(self, cfg: DictConfig, token_usage_callback=None):
        """
        Args:
            cfg: the config
            token_usage_callback: callback for token usage tracking
        """
        self.cfg = cfg
        self.token_usage_callback = token_usage_callback
        
        # Determine model type from model name
        model_name = cfg.llm.model
        self.model_type = self._determine_model_type(model_name)
        
        if self.model_type == "anthropic":
            # Claude via memobench's shared LLM layer (memobench.runtime.get_llm); no client here
            self.client = None
            self.hf_model = None
            self.hf_tokenizer = None
        elif self.model_type == "huggingface":
            # Load local Hugging Face model
            self._load_hf_model(model_name)
            self.client = None
        else:
            # Use OpenAI-compatible client (Azure or standard OpenAI)
            api_type = cfg.openai.get("api_type", "azure")
            if api_type == "openai":
                self.client = get_openai_chat_completion_client(cfg)
            else:
                self.client = get_aoai_chat_completion_client(cfg)
            self.hf_model = None
            self.hf_tokenizer = None
    
    def _determine_model_type(self, model_name: str) -> str:
        """
        Determine if the model is from OpenAI or Hugging Face based on name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            "anthropic" for Claude models, "azure" for OpenAI models, "huggingface" for HF models
        """
        if model_name.lower().startswith(("claude", "gemini")):  # both go through memobench's LLM layer
            return "anthropic"

        # OpenAI/GPT models
        gpt_models = ["gpt-3", "gpt-4", "gpt-5", "o1", "o3"]
        
        if any(gpt_prefix in model_name.lower() for gpt_prefix in gpt_models):
            return "azure"
        
        # Otherwise assume it's a Hugging Face model
        return "huggingface"
    
    def _load_hf_model(self, model_name: str):
        """
        Load Hugging Face model (e.g., Qwen3-8B) locally.
        
        Args:
            model_name: Hugging Face model identifier (e.g., "Qwen3-8B" or "Qwen/Qwen3-8B")
        """
        # Normalize model name to Hugging Face format
        if not "/" in model_name:
            # Convert "Qwen3-8B" to "Qwen/Qwen3-8B"
            if model_name.startswith("Qwen"):
                model_name = f"Qwen/{model_name}"
        
        print(f"Loading Hugging Face model: {model_name}")

        # heavy deps: imported only when a local HF model is actually used
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully on device: {next(self.hf_model.parameters()).device}")

    def invoke(
        self,
        input: Union[str, Sequence[str]],
        prompt_args: Optional[Dict] = None,
        response_format: Any = None,
        source: str = "Unknown",
        **kwargs: Any,
    ) -> str:
        """
        Generate the response from the chat completion model
        Supports:
        - `invoke("Hello {name}", {"name": "Alice"})`
        - `invoke([LLMMessage(role="user", content="Hello!")])`

        if using the reasoning model, add the reasoning parameters such as
            reasoning={
                "effort": "medium",  # low, medium or high
                "summary": "detailed" # auto, concise, or detailed (currently only supported with o4-mini and o3)
            }
        Args:
            input: the input to the model
            prompt_args: the arguments for the prompt template
            response_format: the response format. If None, the response will be a string
            source: the source of the request (used for token usage tracking)
            kwargs: the kwargs for the prompt template

        Returns: the response
        """

        if isinstance(input, str):
            if prompt_args:  # Apply string formatting
                formatted_prompt = input.format(**prompt_args)
            else:
                formatted_prompt = input

            # Create a message with the formatted prompt
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ]

        elif isinstance(input, list):
            messages = input  # Use messages as-is

        else:
            raise ValueError(
                "Input must be a string or a sequence of LLMMessage objects."
            )
        
        # Route to appropriate model
        if self.model_type == "anthropic":
            if source == "Unknown":  # name the calling class for the log (stage comes from response_format)
                caller = inspect.currentframe().f_back.f_locals.get("self", None)
                source = caller.__class__.__name__ if caller is not None else "Unknown"
            return self._invoke_anthropic(messages, response_format, source, **kwargs)
        elif self.model_type == "huggingface":
            return self._invoke_hf(messages, source, **kwargs)
        else:
            return self._invoke_azure(messages, response_format, source, **kwargs)
    
    def _invoke_anthropic(self, messages: list, response_format: Any, source: str, **kwargs) -> Any:
        """Invoke Claude through memobench's shared LLM layer (cache, retries, ledger, request log).

        Returns what _invoke_azure returns: the parsed pydantic object when `response_format` is a
        pydantic class, else the text. Parse failures and max_tokens/refusal stops are appended to
        <runs>/logs/parse_failures.jsonl and answered with upstream's content-filter fallback.
        """
        import anthropic
        from pydantic import BaseModel

        from memobench import runtime
        from memobench.context import get_meta
        from memobench.llm import Req

        max_tokens = kwargs.pop("max_tokens", ANTHROPIC_MAX_TOKENS)
        for name in kwargs:  # temperature, seed, top_p, n, reasoning, ...
            _log_dropped_once(name)
        if response_format is not None and not (
            isinstance(response_format, type) and issubclass(response_format, BaseModel)
        ):
            _log_dropped_once("response_format(non-pydantic)")
            response_format = None

        stage = _stage_for(source, response_format)
        system, anthropic_messages = _to_anthropic_messages(messages)
        anthropic_messages = _apply_cache_split(anthropic_messages, source)

        output_config: Dict[str, Any] = {"effort": ANTHROPIC_EFFORT}
        if response_format is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(response_format),
            }
        params: Dict[str, Any] = {
            "model": self.cfg.llm.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "output_config": output_config,
        }
        if system:
            params["system"] = system

        meta = {**get_meta(), "stage": stage, "source": source}
        result = runtime.get_llm().call(Req(params=params, meta=meta))
        if result.error:
            raise RuntimeError(f"Claude call failed (stage={stage}, source={source}): {result.error}")

        if self.token_usage_callback:
            usage = result.usage or {}
            prompt_tokens = sum(
                usage.get(k) or 0
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            self.token_usage_callback.update(
                prompt_tokens, usage.get("output_tokens") or 0, model=self.cfg.llm.model, source=source
            )

        def fail(error: Exception) -> None:
            _log_parse_failure({
                **meta,
                "raw_text": result.text,
                "error": f"{type(error).__name__}: {error}",
                "stop_reason": result.stop_reason,
                "key": result.key,
            })

        if response_format is None:
            if result.stop_reason in _FALLBACK_STOP_REASONS:
                fail(ValueError(f"stop_reason={result.stop_reason}"))
                if result.stop_reason == "refusal":
                    return ""  # upstream's content-filter result for text calls
            return result.text

        try:
            if result.stop_reason in _FALLBACK_STOP_REASONS:
                raise ValueError(f"stop_reason={result.stop_reason}")
            return response_format.model_validate_json(result.text)
        except Exception as e:
            fail(e)
            return _empty_response(response_format, e)

    def _invoke_hf(self, messages: list, source: str = "Unknown", **kwargs) -> str:
        """Invoke Hugging Face model."""
        import torch

        try:
            # Apply chat template
            text = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.hf_model.device)
            
            # Set seed for reproducibility
            seed = self.cfg.llm.get("seed", 42)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # Get generation parameters
            max_new_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", 512))
            temperature = kwargs.pop("temperature", 0.7)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.hf_model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                )
            
            # Decode the response
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            result = self.hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Token usage tracking (approximate for HF models)
            if self.token_usage_callback:
                prompt_tokens = len(model_inputs.input_ids[0])
                completion_tokens = len(generated_ids[0])
                
                self.token_usage_callback.update(
                    prompt_tokens,
                    completion_tokens,
                    model="qwen3-8b",
                    source=source,
                )
            
            return result
            
        except Exception as e:
            print(f"Error in HF model invocation: {e}")
            raise e
    
    def _invoke_azure(self, messages: list, response_format: Any, source: str, **kwargs) -> str:
        """Invoke Azure OpenAI model."""
        try:

            if response_format:
                response = self.client.beta.chat.completions.parse(
                    messages=messages,
                    model=self.cfg.llm.model,
                    response_format=response_format,
                    seed=self.cfg.llm.get("seed", 42),
                    **kwargs,
                )
            else:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.cfg.llm.model,
                    seed=self.cfg.llm.get("seed", 42),
                    **kwargs,
                )

        except ContentFilterFinishReasonError as e:
            # Azure OpenAI content filter blocked the request
            logger.warning(
                f"Content filter blocked request. Returning empty response. "
                f"Messages length: {len(messages)}, Response format: {response_format.__name__ if response_format else 'None'}"
            )

            # Return an empty response based on the expected format
            if response_format:
                # Create an empty instance of the response format
                # Most Pydantic models used for memory extraction have an 'entries' field
                try:
                    result = response_format(entries=[])
                except Exception:
                    # If that doesn't work, try creating with no arguments
                    try:
                        result = response_format()
                    except Exception:
                        # As a last resort, raise the original error
                        logger.error(f"Could not create empty response for format {response_format}")
                        raise e
            else:
                result = ""

            return result

        except Exception as e:
            print(f"Error in Azure OpenAI invocation: {e}")
            raise e

        if response_format:
            # transform the response to the response format
            result = response.choices[0].message.parsed
        else:
            result = response.choices[0].message.content

        # calculate the tokens
        if self.token_usage_callback:
            usage = response.usage
            assert isinstance(self.token_usage_callback, TokenUsageCallback)

            # update source if not provided
            if source == "Unknown":
                try:
                    caller = inspect.stack()[1].frame.f_locals.get("self", None)
                    source = caller.__class__.__name__ if caller else "Unknown"
                except Exception as e:
                    source = "Unknown"

            self.token_usage_callback.update(
                usage.prompt_tokens,
                usage.completion_tokens,
                model=self.client.model_info["family"],
                source=source,
            )

        return result
