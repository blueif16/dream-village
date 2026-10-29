# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Trajectory scoring for GRPO training.

Implements: J(τ) = w1 * Groundedness - w2 * Redundancy - w3 * Cost
"""

from typing import List, Tuple
from dataclasses import dataclass
import json
import numpy as np
import logging
from omegaconf import DictConfig

from .trajectory_collector import Trajectory

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts (aligned with app/locomo/prompts.py and metrics/llm_judge.py)
# =============================================================================

ANSWER_GENERATION_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question. Some memories may 
    include associated images that provide visual context.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. When images are provided, use the visual information to better understand the context and answer questions
    9. The answer should be less than 5-6 words.


    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. If images are provided, analyze the visual content to add relevant information
    6. Formulate a precise, concise answer based solely on the evidence in the memories and images
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Memories:

    {memories}

    Question: {question}

    Answer:
    """


# Binary accuracy prompt (existing - for backwards compatibility)
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


# NEW: 1-5 Scale Groundedness Prompt for more variance in scoring
ACCURACY_PROMPT_SCALED = """
Your task is to evaluate a generated answer against a gold (ground truth) answer on a scale of 1 to 5.

## SCORING RUBRIC:

**Score 5 - Perfect Answer**
The generated answer is completely correct with all facts accurate. It captures the essential information from the gold answer without any errors in dates, names, locations, quantities, or other factual details.

**Score 4 - Mostly Correct**
The generated answer captures the main point correctly but has minor inaccuracies that don't change the core meaning. Examples:
- Slight date approximations (e.g., "early May" when it was "May 3rd")
- Minor detail variations that preserve the main fact
- Extra information that doesn't contradict the gold answer

**Score 3 - Partially Correct**
The generated answer is partially right but has significant factual errors OR is missing important details. The core topic is identified but key facts are wrong. Examples:
- Correct event but wrong date/time
- Correct person mentioned but wrong action attributed
- Correct location but wrong activity

**Score 2 - Marginally Related**
The generated answer shows some understanding of the topic but most facts are incorrect or the answer is too vague to be useful. Examples:
- Mentions related concepts but gets the specifics wrong
- Identifies the right category but wrong instance
- Heavily incomplete with critical information missing

**Score 1 - Incorrect**
The generated answer is completely wrong, irrelevant, contradicts the gold answer, or provides no useful information. Examples:
- Completely different topic
- "I don't know" or "Insufficient information"
- Factually opposite of the gold answer

## EXAMPLES:

### Example 1: Date/Time Questions
Question: When did I go to Hawaii?
Gold answer: March 2022
- Generated: "March 2022" → Score 5 (exact match)
- Generated: "Spring 2022" → Score 4 (correct timeframe, less specific)
- Generated: "2022" → Score 3 (correct year, missing month)
- Generated: "Last year" → Score 2 (vague, unverifiable)
- Generated: "July 2021" → Score 1 (wrong date)

### Example 2: Person/Name Questions
Question: Who did I meet at the conference?
Gold answer: Dr. Sarah Chen from Stanford
- Generated: "Dr. Sarah Chen from Stanford University" → Score 5 (correct with minor elaboration)
- Generated: "Sarah Chen" → Score 4 (correct person, missing title/affiliation)
- Generated: "A professor from Stanford" → Score 3 (partial info, missing name)
- Generated: "Someone from a university" → Score 2 (too vague)
- Generated: "Dr. Michael Wong from MIT" → Score 1 (completely wrong)

### Example 3: Event/Activity Questions
Question: What did I buy at the farmer's market?
Gold answer: Fresh strawberries and honey
- Generated: "Fresh strawberries and local honey" → Score 5 (correct with minor detail)
- Generated: "Strawberries" → Score 4 (partially complete, missing honey)
- Generated: "Some fruit and honey" → Score 3 (vague on specifics)
- Generated: "Groceries" → Score 2 (too generic)
- Generated: "Flowers and bread" → Score 1 (wrong items)

## YOUR TASK:

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Evaluate the generated answer against the gold answer. Consider:
1. Are the key facts (dates, names, locations, quantities) accurate?
2. Is the core meaning preserved even if phrasing differs?
3. How much useful information is provided?

Return your evaluation as JSON with the following format:
{{
    "reasoning": "Brief explanation of your scoring decision",
    "score": <integer from 1 to 5>
}}
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrajectoryScore:
    """Trajectory-level score components."""
    groundedness: float
    redundancy: float
    cost: float
    total_score: float
    generated_answer: str = ""


# =============================================================================
# Scorer
# =============================================================================

class TrajectoryScorer:
    """
    Scores trajectories for GRPO training.
    
    Scoring function: J(τ) = w1 * Ground(τ) - w2 * Redund(τ) - w3 * Cost(τ)
    
    Groundedness modes:
        - Binary (0/1): Uses ACCURACY_PROMPT with CORRECT/WRONG
        - Scaled (1-5): Uses ACCURACY_PROMPT_SCALED, normalized to [0,1]
        - Soft: Adds semantic similarity component
    
    Redundancy (0 to 1):
        Mean pairwise cosine similarity between memory embeddings
    
    Cost (0 to 1):
        Penalizes steps and RE_QUERY actions
    """
    
    def __init__(
        self,
        cfg: DictConfig = None,
        w_groundedness: float = 1.0,
        w_redundancy: float = 0.3,
        w_cost: float = 0.1,
        max_steps: float = 4.0,
        use_soft_groundedness: bool = True,
        soft_weight: float = 0.3,
        use_scaled_groundedness: bool = False,  # NEW: Use 1-5 scale instead of binary
    ):
        self.cfg = cfg
        self.w1 = w_groundedness
        self.w2 = w_redundancy
        self.w3 = w_cost
        self.max_steps = max_steps
        self.use_soft_groundedness = use_soft_groundedness
        self.soft_weight = soft_weight
        self.use_scaled_groundedness = use_scaled_groundedness  # NEW
        
        self.llm_client = None
        if cfg is not None:
            self._init_llm_client()
            self._init_embedding_client()
    
    # =========================================================================
    # Helper functions
    # =========================================================================

    def _init_llm_client(self):
        """Initialize LLM client for groundedness evaluation."""
        try:
            from memora.utils.llm import get_aoai_chat_completion_client
            self.llm_client = get_aoai_chat_completion_client(self.cfg)
        except Exception as e:
            logger.warning(f"Failed to init LLM client: {e}")
            self.llm_client = None


    def _init_embedding_client(self):
        """Initialize embedding client for redundancy computation."""
        try:
            from memora.utils.embedding import get_aoai_embedding_client
            self.embedding_client = get_aoai_embedding_client(self.cfg)
            self.embedding_model = self.cfg.openai.embedding_model
            logger.info(f"Embedding client initialized: {self.embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to init embedding client: {e}, falling back to word overlap")
            self.embedding_client = None

    def _get_embeddings(self, texts: List[str], max_tokens: int = 8000) -> np.ndarray:
        """Get embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        # Truncate texts to avoid token limits
        truncated = [t[:max_tokens*4] for t in texts]  ## changed here
        
        response = self.embedding_client.embeddings.create(
            input=truncated,
            model=self.embedding_model,
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _llm_call(self, prompt: str, max_tokens: int = 200, use_json: bool = False, for_judging: bool = False) -> str:
        """Make LLM call with error handling."""
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        # Use eval model for judging (to match evaluation pipeline)
        # Use general model for answer generation
        if for_judging:
            model = self.cfg.eval.get("model", self.cfg.openai.model)
        else:
            model = self.cfg.openai.model
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.llm_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    # =========================================================================
    # Groundedness
    # =========================================================================
    
    def _format_memories_for_prompt(self, trajectory: Trajectory) -> str:
        """Format retrieved memories for the answer generation prompt."""
        if not trajectory.retrieved_memories:
            return "(No memories retrieved)"
        
        formatted = []
        for i, mem in enumerate(trajectory.retrieved_memories):
            # value = mem.get('value', '')[:500]
            value = mem.get('value', '')
            timestamp = mem.get('timestamp', 'Unknown')
            formatted.append(f"{timestamp}: {value}")
        
        return "\n".join(formatted)
    
    def _generate_answer(self, trajectory: Trajectory, query: str) -> str:
        """Generate answer from retrieved memories using ANSWER_PROMPT."""
        if not trajectory.retrieved_memories:
            return "Insufficient information."
        
        memories_text = self._format_memories_for_prompt(trajectory)
        
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=query,
            memories=memories_text,
        )
        
        try:
            return self._llm_call(prompt, max_tokens=100)
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            return "Insufficient information."
    
    def _judge_groundedness_scaled(self, query: str, generated_answer: str, ground_truth: str) -> float:
        """
        Compare generated answer to ground truth using 1-5 scale (ACCURACY_PROMPT_SCALED).
        
        Returns:
            Score normalized to [0, 1]: (raw_score - 1) / 4
            - Score 1 → 0.0
            - Score 2 → 0.25
            - Score 3 → 0.5
            - Score 4 → 0.75
            - Score 5 → 1.0
        """
        if not ground_truth:
            logger.debug("No ground truth provided, defaulting to 0.0")
            return 0.0
        
        prompt = ACCURACY_PROMPT_SCALED.format(
            question=query,
            gold_answer=ground_truth,
            generated_answer=generated_answer,
        )
        
        try:
            response = self._llm_call(prompt, max_tokens=200, use_json=True, for_judging=True)
            parsed = json.loads(self._extract_json(response))
            raw_score = parsed.get("score", 1)
            
            # Validate score is in valid range
            raw_score = max(1, min(5, int(raw_score)))
            
            # Normalize to [0, 1]
            normalized = (raw_score - 1) / 4.0
            
            logger.debug(f"Scaled groundedness: raw={raw_score}, normalized={normalized:.2f}, reasoning={parsed.get('reasoning', 'N/A')[:50]}")
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Scaled groundedness judging failed: {e}")
            return 0.0

    def _judge_groundedness(self, query: str, generated_answer: str, ground_truth: str) -> float:
        """Compare generated answer to ground truth using ACCURACY_PROMPT (0/1 score)."""
        if not ground_truth:
            logger.debug("No ground truth provided, defaulting to 0.0")
            return 0.0
        
        prompt = ACCURACY_PROMPT.format(
            question=query,
            gold_answer=ground_truth,
            generated_answer=generated_answer,
        )
        
        try:
            response = self._llm_call(prompt, max_tokens=150, use_json=True, for_judging=True)
            parsed = json.loads(self._extract_json(response))
            label = parsed.get("label", "WRONG")
            return 1.0 if label == "CORRECT" else 0.0
        except Exception as e:
            logger.warning(f"Groundedness judging failed: {e}")
            return 0.0

    def _compute_soft_groundedness(self, generated: str, gold: str) -> float:
        """Compute soft groundedness using embedding similarity.
        
        Returns similarity score in [0, 1].
        """
        if not generated or not gold:
            return 0.0
        
        if self.embedding_client is None:
            logger.warning("Embedding client not available for soft groundedness")
            return 0.0
            
        try:
            embeddings = self._get_embeddings([generated, gold])
            if len(embeddings) < 2:
                return 0.0
            
            sim = self._cosine_similarity(embeddings[0], embeddings[1])
            # Normalize from [-1, 1] to [0, 1]
            return (sim + 1) / 2
        except Exception as e:
            logger.warning(f"Soft groundedness computation failed: {e}")
            return 0.0
    
    def compute_groundedness(self, trajectory: Trajectory, query: str, ground_truth: str) -> Tuple[float, str]:
        """
        Compute groundedness score.
        
        Modes (in order of precedence):
        
        1. If use_scaled_groundedness=True:
            Uses 1-5 scale scoring, normalized to [0, 1]
            If use_soft_groundedness also True, combines:
                (1-soft_weight)*scaled + soft_weight*soft
        
        2. If use_soft_groundedness=True (and scaled=False):
            Returns weighted combination: (1-soft_weight)*binary + soft_weight*soft
            This provides gradient signal even when binary=0
        
        3. Default (both False):
            Returns binary 0.0 or 1.0 based on LLM judge
        
        Returns:
            (groundedness_score, generated_answer)
        """
        if not trajectory.retrieved_memories:
            return 0.0, "Insufficient information."
        
        generated_answer = self._generate_answer(trajectory, query)
        
        # Determine base score (scaled or binary)
        if self.use_scaled_groundedness:
            base_score = self._judge_groundedness_scaled(query, generated_answer, ground_truth)
        else:
            base_score = self._judge_groundedness(query, generated_answer, ground_truth)
        
        # Apply soft component if enabled
        if not self.use_soft_groundedness:
            return base_score, generated_answer
        
        # Hybrid scoring: base (scaled or binary) + soft component
        soft_score = self._compute_soft_groundedness(generated_answer, ground_truth)
        
        # Weighted combination
        # groundedness = (1 - self.soft_weight) * base_score + self.soft_weight * soft_score

        groundedness = base_score
        
        score_type = "scaled" if self.use_scaled_groundedness else "binary"
        logger.debug(f"Groundedness: {score_type}={base_score:.2f}, soft={soft_score:.3f}, combined={groundedness:.3f}")
        
        return groundedness, generated_answer

    # =========================================================================
    # Redundancy
    # =========================================================================
    
    def compute_redundancy(self, trajectory: Trajectory) -> float:
        """
        Compute redundancy via mean pairwise cosine similarity of embeddings.
        
        Falls back to Jaccard word overlap if embedding client is unavailable.
        Returns:
            Score in [0, 1]. Higher = more redundant = worse.
        """
        memories = trajectory.retrieved_memories
        if len(memories) <= 1:
            return 0.0

        # Try embedding-based similarity first
        if self.embedding_client is not None:
            try:
                return self._compute_redundancy_embedding(memories)
            except Exception as e:
                logger.warning(f"Embedding redundancy failed: {e}, falling back to word overlap")
        
        # Fallback to word overlap
        return self._compute_redundancy_jaccard(memories)

    def _compute_redundancy_embedding(self, memories: List[dict]) -> float:
        """
        Compute redundancy using embedding cosine similarity.
        
        Returns mean pairwise cosine similarity in [0, 1].
        """
        # Extract memory texts
        texts = [m.get("value", "") for m in memories if m.get("value")]
        
        if len(texts) <= 1:
            return 0.0
        
        # Get embeddings
        embeddings = self._get_embeddings(texts)
        
        if len(embeddings) <= 1:
            return 0.0
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                # Cosine similarity is in [-1, 1], normalize to [0, 1]
                sim_normalized = (sim + 1) / 2
                similarities.append(sim_normalized)
        
        return float(np.mean(similarities)) if similarities else 0.0


    def _compute_redundancy_jaccard(self, memories: List[dict]) -> float:
        """
        Fallback: Compute redundancy via mean pairwise Jaccard similarity.
        """
        # Extract word sets
        word_sets = [
            set(m.get("value", "").lower().split())
            for m in memories
        ]
        
        # Pairwise Jaccard
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                s1, s2 = word_sets[i], word_sets[j]
                if s1 and s2:
                    similarities.append(len(s1 & s2) / len(s1 | s2))
        
        return float(np.mean(similarities)) if similarities else 0.0
        
    # =========================================================================
    # Cost
    # =========================================================================
    
    def compute_cost(self, trajectory: Trajectory) -> float:
        """
        Compute cost based on steps and RE_QUERY actions.
        
        Returns:
            Score in [0, 1]. Higher = more expensive = worse.
        """
        num_steps = len(trajectory.steps)
        num_requery = sum(1 for s in trajectory.steps if s.action == "RE_QUERY")
        
        step_cost = num_steps / self.max_steps
        requery_penalty = num_requery * 0.2
        
        return min(1.0, step_cost + requery_penalty)

    # =========================================================================
    # Main Scoring
    # =========================================================================
    
    def score_trajectory(
        self, 
        trajectory: Trajectory,
        query: str = None,
        ground_truth: str = None,
    ) -> TrajectoryScore:
        """
        Compute full trajectory score: J(tau) = w1*G - w2*R - w3*C
        
        Args:
            trajectory: Trajectory to score
            query: The query string (required for groundedness)
            ground_truth: Ground truth answer (required for groundedness)
        """
        groundedness, generated_answer = self.compute_groundedness(
            trajectory, query or "", ground_truth or ""
        )
        redundancy = self.compute_redundancy(trajectory)
        cost = self.compute_cost(trajectory)
        
        total = self.w1 * groundedness - self.w2 * redundancy - self.w3 * cost
        
        # Update trajectory object
        trajectory.groundedness = groundedness
        trajectory.redundancy = redundancy
        trajectory.cost = cost
        trajectory.trajectory_score = total
        trajectory.generated_answer = generated_answer
        
        return TrajectoryScore(
            groundedness=groundedness,
            redundancy=redundancy,
            cost=cost,
            total_score=total,
            generated_answer=generated_answer,
        )
    
    def score_trajectories(self, trajectories: List[Trajectory]) -> List[TrajectoryScore]:
        """Score multiple trajectories."""
        scores = []
        for idx, traj in enumerate(trajectories):
            logger.debug(f"Scoring trajectory {idx+1}/{len(trajectories)}")
            score = self.score_trajectory(traj)
            scores.append(score)
            logger.debug(
                f"  G={score.groundedness:.0f}, R={score.redundancy:.3f}, "
                f"C={score.cost:.3f}, Total={score.total_score:.3f}"
            )
        return scores