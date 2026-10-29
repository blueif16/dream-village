# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Local LLM Policy Retriever for Memory Search.

Uses a local open-source model (e.g., Qwen) as the policy LLM instead of closed-source models.
Serves as a baseline for comparison with RL-trained retrieval policies.

Two modes:
1. Baseline mode: Untrained model (no checkpoint)
2. Fine-tuned mode: Load GRPO-trained checkpoint for evaluation
"""

import json
import logging
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from memora.retriever.base_retriever import BaseMemoryRetriever
from memora.retriever.policy_utils import (
    POLICY_SYSTEM_MESSAGE,
    format_user_message,
    select_from_frontier,
    parse_json_response,
    validate_policy_decision,
)
from memora.core.memory_expander import MemoryExpander
from memora.core.memory_entry import MemoryEntry
from memora.core.memory import AgentMemory
from memora.utils.memory import dedup_memories
from memora.core.memory import AgentMemory, QueryMode 

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODEL CACHE - ensures only one model is loaded across all instances
# ============================================================================
_LOCAL_MODEL_CACHE: Dict[str, Any] = {}  

def _get_or_load_model(
    model_name: str,
    checkpoint_path: Optional[str],
    device: str,
    load_in_8bit: bool,
) -> tuple:
    """
    Get model from global cache or load it.
    Ensures only ONE model instance exists in memory.
    """
    cache_key = f"{model_name}_{checkpoint_path or 'baseline'}"
    
    if cache_key in _LOCAL_MODEL_CACHE:
        logger.info(f"Using cached model: {cache_key}")
        return _LOCAL_MODEL_CACHE[cache_key]
    
    logger.info(f"Loading local model: {model_name} (checkpoint: {checkpoint_path})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build load kwargs
    load_kwargs = {
        "trust_remote_code": True,
        "use_cache": True,
    }
    
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
    )
    
    # Load LoRA checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=False,
        )
        logger.info("✓ LoRA checkpoint loaded")
    
    model.eval()
    
    # Cache it
    _LOCAL_MODEL_CACHE[cache_key] = (model, tokenizer)
    logger.info(f"Model cached: {cache_key}")
    
    return model, tokenizer


class LocalPolicyRetriever(BaseMemoryRetriever):
    """
    Local LLM-based iterative memory retrieval.
    
    Uses a local open-source model for policy decisions instead of GPT-4.
    This provides a baseline for comparing with RL-trained policies.
    
    Modes:
    - Baseline: Set checkpoint_path=None for untrained baseline
    - Fine-tuned: Set checkpoint_path to GRPO-trained checkpoint
    
    Example:
        # Baseline mode
        retriever = LocalPolicyRetriever(cfg, memory_client=memory)
        
        # Fine-tuned mode (after GRPO training)
        retriever = LocalPolicyRetriever(
            cfg, 
            memory_client=memory,
            checkpoint_path="./grpo_retrieval_output/final"
        )
    """

    def __init__(
        self,
        cfg: DictConfig,
        memory_client: Optional[AgentMemory] = None,
        model_name: str = None,
        max_steps: int = None,
        device: str = None,
        checkpoint_path: str = None,
        temperature: float = None,
        load_in_8bit: bool = None,
    ):
        """
        Initialize Qwen policy retriever.
        
        Args:
            cfg: Configuration object
            memory_client: Pre-initialized memory client
            model_name: Base Qwen model name (reads from config if None)
            max_steps: Maximum retrieval iterations (reads from config if None)
            device: Device to use (auto-detected if None)
            checkpoint_path: Path to GRPO fine-tuned checkpoint (None for baseline)
            temperature: Generation temperature (reads from config if None)
            load_in_8bit: Use 8-bit quantization (reads from config if None)
        """
        super().__init__(cfg)
        self.memory_client = memory_client
        
        # Read from config with fallbacks
        local_cfg = cfg.get("retrieval", {}).get("local_policy", {})
        
        self.model_name = model_name or local_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.max_steps = max_steps or local_cfg.get("max_steps", 5)
        self.temperature = temperature if temperature is not None else local_cfg.get("temperature", 1.0)
        self.load_in_8bit = load_in_8bit if load_in_8bit is not None else local_cfg.get("load_in_8bit", False)
        self.checkpoint_path = checkpoint_path or local_cfg.get("checkpoint_path", None)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None

        if cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY
            
        enable_relaxed_frontier = self.cfg.retrieval.get("enable_relaxed_frontier", False)
        max_workers = self.cfg.eval.get("max_workers", 5)
        
        self.expander = MemoryExpander(
            memory_client=memory_client,
            enable_relaxed_frontier=enable_relaxed_frontier,
            max_workers=max_workers
        )

        self.last_trace = []
        
        # Mode indicator
        self.mode = "fine-tuned" if self.checkpoint_path else "baseline"
        logger.info(f"LocalPolicyRetriever initialized: model={self.model_name}, mode={self.mode}")

    def _load_model(self):
        """Load model from global cache"""
        if self.model is not None:
            return

        self.model, self.tokenizer = _get_or_load_model(
            model_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            load_in_8bit=self.load_in_8bit,
        )

    def _call_policy(
        self,
        user_question: str,
        current_query: str,
        working_set: List[MemoryEntry],
        frontier: Dict[str, MemoryEntry],
        step: int,
    ) -> Dict[str, Any]:
        """
        Call Qwen model to decide the next action.
        
        Returns:
            Decision dict with action, reason, confidence, frontier_ids, new_query
        """
        self._load_model()
        
        # Build prompt using shared formatter
        user_message = format_user_message(
            user_question=user_question,
            current_query=current_query,
            working_set=working_set,
            frontier=frontier,
            step=step,
            max_steps=self.max_steps,
        )
        
        messages = [
            {"role": "system", "content": POLICY_SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]
        
        # Tokenize
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=300,   ## 256
                # temperature=self.temperature if self.temperature > 0 else 1.0,
                # do_sample=self.temperature > 0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_ids = outputs[0, prompt_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Parse and validate
        decision = parse_json_response(response)
        decision = validate_policy_decision(decision, frontier)
        
        return decision

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_hybrid_search: bool = None,
        latency_tracker = None,  # ADD THIS
        **kwargs
    ) -> List[MemoryEntry]:
        """
        Retrieve memories using Qwen-guided iterative search.
        
        Args:
            query: Natural language query
            top_k: Number of initial results
            enable_hybrid_search: Use hybrid search
            latency_tracker: Optional LatencyTracker for performance measurement
            
        Returns:
            List of retrieved MemoryEntry objects
        """
        import time  # Add if not already imported at top
        
        self.expander.reset()
        self.last_trace = []
        
        if top_k is None:
            top_k = self.cfg.memory.get("top_k", 10)
        if enable_hybrid_search is None:
            enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)

        # Step 0: Initial retrieval
        step_start = time.time()
        
        memory_entries = self.memory_client.query(
            query,
            top_k=top_k,
            enable_hybrid_search=enable_hybrid_search,
            query_mode=self.query_mode,
            latency_tracker=latency_tracker,
        )
        
        current_query = query
        frontier: Dict[str, MemoryEntry] = {}
        frontier = self.expander.build_frontier(frontier, memory_entries)
        
        step_duration = time.time() - step_start
        
        step_data = {
            "step": 0,
            "action": "INIT_RETRIEVE",
            "query": query,
            "memories_count": len(memory_entries),
            "frontier_size": len(frontier),
            "duration": step_duration,
        }
        
        # Add search component breakdown from tracker if available
        if latency_tracker:
            search_breakdown = latency_tracker.get_search_breakdown()
            if search_breakdown:
                step_data["search_components"] = search_breakdown
            latency_tracker.add_retrieval_step(step_data)
        
        self.last_trace.append(step_data)
        
        logger.info(f"Initial retrieval: {len(memory_entries)} memories, {len(frontier)} frontier")

        # Iterative retrieval loop
        for step_idx in range(1, self.max_steps + 1):
            step_start = time.time()
            llm_start = time.time()
            
            decision = self._call_policy(
                user_question=query,
                current_query=current_query,
                working_set=memory_entries,
                frontier=frontier,
                step=step_idx,
            )
            
            llm_duration = time.time() - llm_start
            
            # Track policy LLM time
            if latency_tracker:
                latency_tracker.add_timing("policy_llm", llm_duration)
            
            action = decision.get("action", "STOP")
            
            # STOP
            if action == "STOP":
                step_duration = time.time() - step_start
                step_data = {
                    "step": step_idx,
                    "action": "STOP",
                    "reason": decision.get("reason", ""),
                    "confidence": decision.get("confidence", 0.0),
                    "duration": step_duration,
                    "llm_duration": llm_duration,
                }
                self.last_trace.append(step_data)
                
                if latency_tracker:
                    latency_tracker.add_retrieval_step(step_data)
                
                logger.info(f"Step {step_idx}: STOP - {decision.get('reason', '')}")
                break
            
            # EXPAND
            if action == "EXPAND":
                frontier_ids = decision.get("frontier_ids", [])
                chosen = select_from_frontier(frontier, frontier_ids)
                
                if chosen:
                    memory_entries = dedup_memories(memory_entries + chosen)
                    for mem in chosen:
                        frontier.pop(mem.index, None)
                    frontier = self.expander.build_frontier(frontier, chosen)
                
                step_duration = time.time() - step_start
                step_data = {
                    "step": step_idx,
                    "action": "EXPAND",
                    "added": len(chosen),
                    "frontier_ids": frontier_ids,
                    "new_frontier_size": len(frontier),
                    "duration": step_duration,
                    "llm_duration": llm_duration,
                }
                self.last_trace.append(step_data)
                
                if latency_tracker:
                    latency_tracker.add_retrieval_step(step_data)
                
                logger.info(f"Step {step_idx}: EXPAND - added {len(chosen)} memories")
                continue
            
            # RE_QUERY
            if action == "RE_QUERY":
                new_query = decision.get("new_query", current_query)
                current_query = new_query
                
                new_entries = self.memory_client.query(
                    current_query,
                    top_k=top_k,
                    enable_hybrid_search=enable_hybrid_search,
                    query_mode=self.query_mode,
                    latency_tracker=latency_tracker,
                )
                
                memory_entries = dedup_memories(memory_entries + new_entries)
                frontier = self.expander.build_frontier(frontier, new_entries)
                
                step_duration = time.time() - step_start
                step_data = {
                    "step": step_idx,
                    "action": "RE_QUERY",
                    "new_query": new_query,
                    "new_memories": len(new_entries),
                    "total_memories": len(memory_entries),
                    "duration": step_duration,
                    "llm_duration": llm_duration,
                }
                
                # Add search component breakdown from tracker if available
                if latency_tracker:
                    search_breakdown = latency_tracker.get_search_breakdown()
                    if search_breakdown:
                        step_data["search_components"] = search_breakdown
                    latency_tracker.add_retrieval_step(step_data)
                
                self.last_trace.append(step_data)
                
                logger.info(f"Step {step_idx}: RE_QUERY - '{new_query}' got {len(new_entries)} new")
                continue
            
            # Invalid action - stop
            step_duration = time.time() - step_start
            step_data = {
                "step": step_idx,
                "action": "INVALID",
                "original_action": action,
                "duration": step_duration,
            }
            self.last_trace.append(step_data)
            
            if latency_tracker:
                latency_tracker.add_retrieval_step(step_data)
            
            logger.warning(f"Step {step_idx}: Invalid action '{action}', stopping")
            break
        
        logger.info(f"Retrieval complete: {len(memory_entries)} memories in {len(self.last_trace)} steps")
        return memory_entries

    def get_trace(self) -> List[Dict]:
        """Return trace from last retrieval for analysis."""
        return self.last_trace
    
    def get_mode(self) -> str:
        """Return current mode: 'baseline' or 'fine-tuned'."""
        return self.mode