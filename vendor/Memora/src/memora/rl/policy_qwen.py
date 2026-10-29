# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .policy import BasePolicy, PolicyState, PolicyOutput
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

'''
References:
1. Chat Template: https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate
'''

class QwenPolicy(BasePolicy):
    """
    Decoder-style policy using Qwen model.
    Generates JSON actions and computes token-level log probabilities.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = None,
        load_in_8bit: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",  # Important for batched generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Build load kwargs
        load_kwargs = {
            "trust_remote_code": True,
            "use_cache": False,  # Disable for training compatibility
        }
        
        # Quantization config
        if self.load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
            if self.device == "cuda":
                load_kwargs["device_map"] = {"": 0}
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
                    "gate_proj", "up_proj", "down_proj"       # MLP
                ],
                bias="none",
                inference_mode=False,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()

            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        self._loaded = True
        logger.info(f"Model loaded on {self.device}, LoRA={self.use_lora}")

    
    def _prepare_inputs(
        self,
        state: PolicyState,
        action_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Prepare tokenized inputs from state and action.
        
        Returns:
            Tuple of (input_ids, attention_mask, prompt_len)
        """
        # === Tokenize prompt (system + user) ===
        prompt_messages = self.build_messages(state)
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,  # including the "assistant" prefix
            return_tensors="pt"
        )
        # prompt_ids.input_ids: [1, prompt_len]
        prompt_len = prompt_ids.shape[1]  
        

        # === Tokenize complete sequence (system + user + assistant response) ===
        complete_messages = self.build_messages_with_response(state, action_text)
        complete_ids = self.tokenizer.apply_chat_template(
            complete_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(self.model.device)
        
        attention_mask = torch.ones_like(complete_ids)
        return complete_ids, attention_mask, prompt_len
    

    def get_action_logprobs(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Compute log probabilities of action tokens 
        Args:
            input_ids: complete sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            prompt_lens: Lengths of prompt for each sample [batch_size]
        """
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # outputs.logits: [batch_size, seq_len, vocab_size]
        batch_size = input_ids.shape[0]
        action_logprobs = []

        for i in range(batch_size):
            prompt_len = int(prompt_lens[i].item())
            seq_len = int(attention_mask[i].sum().item())

            # === Extract logits for action tokens ===
            # logits[t] predicts token at position t+1

            # action_logits -> [action_len, vocab_size]
            action_logits = outputs.logits[i, prompt_len-1: seq_len-1, :]

            # action_ids -> [action_len]
            action_ids = input_ids[i, prompt_len: seq_len]

            # === Compute log probabilities ===
            # [action_len, vocab_size]
            log_probs = F.log_softmax(action_logits, dim=-1)

            # === Compute log probabilities of actual action tokens ===
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=action_ids.unsqueeze(-1)
            ).squeeze(-1)   # [action_len]

            action_logprobs.append(token_log_probs.sum())
        
        return torch.stack(action_logprobs)  # [batch_size]

    def compute_log_prob(
        self,
        states: List[PolicyState],
        action_texts: List[str],
    ) -> torch.Tensor:
        """
        Batched logprob computation
        Args:
            states: List of PolicyState object
            action_texts: List of action JSON strings
        
        Returns: 
            Tensor [batch_size] with log pi(action|state) for each sample
        """
        
        self._load_model()

        if(len(states) == 0):
            return torch.tensor([], device=self.device)
    
        # === Step 1: Tokenize all samples ===
        all_input_ids = []
        all_prompt_lens = []

        for state, action_text in zip(states, action_texts):
            input_ids, _, prompt_len = self._prepare_inputs(state, action_text)
            all_input_ids.append(input_ids.squeeze(0))  # [seq_len], Remove batch dim
            all_prompt_lens.append(prompt_len)
        
        # == Step 2: Pad to same length ===
        # pad_sequence pads shorted sequences to match the longest
        # Result: [batch_size, max_seq_len]
        padded_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Create attention mask: 1 for real token, 0 for padding
        attention_mask = (padded_ids != self.tokenizer.pad_token_id).long()
        prompt_lens = torch.tensor(all_prompt_lens, device=padded_ids.device)

        # === Step 3: Single forward pass ===
        outputs = self.model(
            input_ids=padded_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # output.logits: [batch_size, max_seq_len, vocab_size]
        
        # === Step 4: Compute logprobs for each sample ===
        batch_size = padded_ids.shape[0]
        logprobs = []

        for i in range(batch_size):
            prompt_len = int(prompt_lens[i].item())
            seq_len = int(attention_mask[i].sum().item())

            # Logits for predicting action tokens
            # logits[t] predicts token at position t+1

            action_logits = outputs.logits[i, prompt_len-1:seq_len-1, :]  # [action_len, vocab]
            action_ids = padded_ids[i, prompt_len:seq_len]                # [action_len]

            # Log softmax over vocabulary
            log_probs = F.log_softmax(action_logits, dim=-1)  # [action_len, vocab]

            # Gather log-probs of actual tokens
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=action_ids.unsqueeze(-1),  # [action_len, 1]
            ).squeeze(-1)  # [action_len]

            logprobs.append(token_log_probs.sum())

        return torch.stack(logprobs)  # [batch_size]

    def select_action(
        self,
        state: PolicyState,
        temperature: float = 1.0,
        do_sample: bool = True,
        training_step: int = None,  # NEW
    ) -> PolicyOutput:
        """
        Rollout-time action sampling

        Outputs: 
        - 'action_text': the sampled JSON text (the action)
        - 'log_prob': (pi_old) stored in trajectories and later used in the GRPO ratio

        Notes:
        - The function runs generate() under torch.no_grad() since rollouts do not require gradients.
        - Compute old_logprob by running a forward pass on the exact sampled tokens (outputs.sequences).
          This guarantees consistency with training-time logprob computation and avoids any generation-time logits processors/warpers.
        """
        self._load_model()
        
        # === Build prompt (chat template) ===
        messages = self.build_messages(state)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,              
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # prompt_ids: [1, prompt_len]
        prompt_attention_mask = torch.ones_like(prompt_ids)
        prompt_len = prompt_ids.shape[1]
        
        # === Sample / generate ===
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,              
                max_new_tokens=300,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=0.9 if do_sample else 1.0,
                return_dict_in_generate=True,
                output_scores=False,            # we do not rely on scores for logprob
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # outputs.sequences: [1, prompt_len + generated_len]
        full_ids = outputs.sequences
        full_attention_mask = torch.ones_like(full_ids)
        
        # Continuation tokens only 
        generated_ids = full_ids[0, prompt_len:] 
        action_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)


        # === Compute old log prob: log pi_old(a|s) ===
        # Used later in training in the ratio: log pi_theta(a|s) - log pi_old(a|s)
        with torch.no_grad():
            old_logprob = self.get_action_logprobs(
                input_ids=full_ids,                   # [1, prompt_len + gen_len]
                attention_mask=full_attention_mask,   # [1, prompt_len + gen_len]
                prompt_lens=torch.tensor([prompt_len], device=full_ids.device),
            )
        
        # Parse JSON action from generated text
        # Validates the generated JSON text, which is used while training to compute log pi_theta(a|s)
        result = self._parse_action(action_text, state.frontier, training_step=training_step)

        # Important: This is old log prob for the sampled action
        result.log_prob = float(old_logprob.item())
        return result

    # ========================================================================
    # Model management methods
    # ========================================================================
    
    def get_model(self):
        """Get the underlying model (for optimizer setup)."""
        self._load_model()
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        self._load_model()
        return self.tokenizer
    
    def train_mode(self):
        """Set model to training mode."""
        self._load_model()
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self._load_model()
        self.model.eval()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self._load_model()
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._loaded = True
        logger.info(f"Loaded checkpoint from {path}")