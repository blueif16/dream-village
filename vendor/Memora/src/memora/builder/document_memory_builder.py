# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import List, Optional

from memora.core.memory_entry import MemoryEntry
from typing import Dict, List, Optional
from omegaconf import DictConfig
from memora.core.memory_entry import MemoryEntry
from memora.utils.llm import ChatCompletionModel
from memora.utils.memory import (
    convert_memory_output,
)
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig
from memora.core.memory import AgentMemory
import logging

from memora.builder.memory_builder import MemoryBuilder, MemoryOutputs

# Initialize module logger
logger = logging.getLogger(__name__)

PROMPT_BUILD_DOCUMENT_MEMORY = """
You are an expert assistant for knowledge extraction and memory construction. 
Your task is to extract both factual and procedural memories from a given document segment 
and represent them as memory entries for a long-term memory database.

=== Output Format ===
Each extracted memory should follow one of these formats:

Factual Memory:
MemType: Factual
MemIndex: <short, semantically rich title summarizing the key concept>
MemValue: <detailed factual content; preserve all informative details>

Procedural Memory:
MemType: Procedural
MemIndex: <concise, descriptive title of the process or workflow>
MemSteps:
1. <first step>
2. <second step>
3. ...
Summary: <brief description of the procedure’s purpose, outcome, or trigger condition>

=== Guidelines ===
1. Fidelity & Granularity
   - Preserve as much detail as possible from the segment.
   - Use the entire segment as one memory value when coherent.
   - Split only if the text clearly contains distinct, independent ideas or processes.
   - Never over-summarize; keep full definitions, examples, and conditions.

2. Detecting Procedural Memories
   - Classify as procedural if the text describes ordered steps, actions, or decision logic.
   - Common cues: “first”, “then”, “follow”, “process”, “step”, “if/when”, “procedure”.
   - Represent each action as a numbered MemStep.
   - Provide a Summary explaining what the procedure accomplishes.

3. MemIndex Construction
   - 3–8 words, compact but meaningful.
   - Include contextual or hierarchical qualifiers if needed (e.g., “System Recovery > Validation Step”).
   - Avoid vague titles like “Overview” or “Details” without context.

4. Factual Memory Writing
   - Use neutral, factual sentences.
   - Include definitions, metrics, examples, or role responsibilities if available.
   - Avoid meta phrases like “This section describes…”.

5. Procedural Memory Writing
   - List steps in order.
   - Make each step actionable and clear.
   - Include conditional logic if mentioned.
   - Keep Summary concise but descriptive.

6. Context Awareness
   - If the document has hierarchy (e.g., Document > Section > Subsection),
     reflect that context implicitly in the MemIndex.
   - Example: “Incident Management > Stage 2 - Assessment” 
     → MemIndex: “Incident Response - Stage 2 Assessment Process”.

=== Example ===
Input:
Incident Escalation Process
1. Detect potential outage and validate telemetry.
2. Notify incident commander.
3. Open communication bridge.
4. Escalate to leadership if impact persists beyond 30 minutes.

Output:
MemType: Procedural
MemIndex: Incident Escalation Process
MemSteps:
1. Detect potential outage and validate telemetry.
2. Notify the incident commander.
3. Open a communication bridge.
4. Escalate to leadership if the impact persists beyond 30 minutes.
Summary: A four-step escalation workflow ensuring timely leadership engagement when impact persists.

=== Final Instruction ===
Process the following segment:
{segment_content}

Generate the appropriate Factual or Procedural memory entries 
following the above format and guidelines.

"""

class DocumentMemoryBuilder(MemoryBuilder):

    def __init__(self, cfg: DictConfig, agent_memory: AgentMemory, model_client: ChatCompletionModel):
        super().__init__(cfg, agent_memory, model_client)
            

    def generate_memory_entries(
        self, content: Union[str, Dict], metadata: Optional[Dict], enable_cue_index: bool = False
    ) -> List[MemoryEntry]:
        """
        Build memory from content.

        Args:
            content: The content to extract memories from
            metadata: Additional metadata to associate with the memory entries
            enable_cue_index: If True, generate cue indices in a separate LLM call. If False, use original format.

        Returns:
            memory entries extracted from the content
        """

        # Step 1: Always use PROMPT_BUILD_DOCUMENT_MEMORY to extract memories
        build_memory_prompt = PROMPT_BUILD_DOCUMENT_MEMORY
        response_format = MemoryOutputs

        # handle multimodal content (dict with text and images keys)
        memories = self.handle_multimodal_content(content, metadata, build_memory_prompt, response_format)
        
        if not memories:
            # handle pure text content
            memories = self._model_client.invoke(
                input=build_memory_prompt,
                prompt_args={
                    "segment_content": content,
                },
                response_format=response_format,
            )
        
        # Step 2: Convert to memory entries (without cue indices yet)
        memory_entries = convert_memory_output(memories, metadata, enable_cue_index=False)
        
        # Step 3: If cue index is enabled, generate cue indices for all memories in a single batch call
        if enable_cue_index and memory_entries:
            try:
                # Prepare batch input
                memories_batch = [
                    {"index": entry.index, "value": entry.value}
                    for entry in memory_entries
                ]
                
                # Generate cue indices for all memories in one LLM call
                cue_indices_map = self.cue_index_generator.generate_cue_indices_batch(memories_batch)
                
                # Attach cue indices to each entry
                for entry in memory_entries:
                    cue_indices = cue_indices_map.get(entry.index, [])
                    # Only set cue_indices if the list is non-empty
                    entry.cue_indices = "||".join(cue_indices) if cue_indices and len(cue_indices) > 0 else ""
                    
            except Exception as e:
                logger.warning(f"Failed to generate cue indices in batch: {e}")
                # Fall back to empty cue indices
                for entry in memory_entries:
                    entry.cue_indices = ""
        
        return memory_entries

    def generate_episodic_memory(
        self, content: Union[str, Dict], metadata: Optional[Dict]
    ) -> Optional[MemoryEntry]:
        """
        Document memory builder does not support episodic memory.
        
        Returns:
            None
        """
        return None
