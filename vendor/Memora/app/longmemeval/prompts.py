# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

ANSWER_PROMPT_NEMORI_LONGMEMEVAL = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a user and an AI assistant. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both the user and the AI assistant
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
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

   {{ memories }}

    Question: {{ question }}
    Question Date: {{ question_date }}

    Answer:
"""


ANSWER_PROMPT_LONGMEMEVAL = """
You are an intelligent Memory Assistant. Your task is to answer the user's question accurately by reasoning from conversational memories.

# TASK
Your are provided with a user's question and a set of timestamped memories.
You must synthesize information from the memories to provide a accurate and concise answer to the user's question. 
This may require multi-hop reasoning, connecting dots across different memories, and performing time calculations when necessary.
If the memories do not contain sufficient information to answer the question, explicitly state that.

# GUIDELINES
1. **Answer Questions:** Answer questions accordingly to the memories provided.
2. **Chain-of-Thought:** Use a structured Chain-of-Thought approach to ensure no details are missed.
3. **Complete Entity Retention:** Always include full names, locations, brands, and other proper nouns as they appear in the memories. Never use generic terms like "a colleague" or "the company."
4. **Absolute Precision:** Retain all exact numbers, prices, dates (calculate relative ones), frequencies, and proper nouns.
5. **Multi-Hop Reasoning:** You must actively connect dots. If Memory A says "User works at Acme Corp" and Memory B says "Acme Corp announced layoffs," you must infer the user might be affected. Do not treat memories in isolation.
6. **Time Calculations:** Always calculate the actual dates from relative references.

# REASONING PROCESS (Think step-by-step):

Please follow the following Chain-of-Thought process step-by-step to ensure a comprehensive and accurate answer:

## STEP 1: EVIDENCE EXTRACTION
Identify all memories that contain information relevant to the question.
   - **Identify:** Quote the specific memory content and its timestamp.
   - **Extract Entities:** List all specific entities (People, Locations, Objects) found in these snippets.

## STEP 2: LOGICAL SYNTHESIS & MULTI-HOP REASONING
Analyze the evidence to establish connections and derive new insights.
- **Link:** Identify connections between different memories (e.g., "Memory 1 mentions X, Memory 2 links X to Y, therefore...").
- **Cross-reference:** Cross-reference entities appearing in multiple memories to uncover implicit relationships.
- **Infer:** Perform logical inferences where direct evidence is lacking but strongly suggested by the data.

## STEP 3: RESOLVE ENTITIES & TIME CALCULATIONS
Resolve any ambiguous reference to entities or timeframes to ensure clarity.
- **Resolve:** For any ambiguous entities (e.g., "the colleague," "the project"), clarify who/what they refer to based on context. For example, instead of "the project" specify "Project Phoenix"; instead of "a gift," specify "the birthday gift from Sarah" if that is clear from the memories.
- **Calculate Time:** For any relative time references (e.g., "last month," "in two weeks"), compute the absolute date based on the memory timestamp and the question date. For example, if a memory from 15 March 2023 mentions "tomorrow," then the date is 16 March 2023.

## STEP 4: REFUTATION OF CONTRADICTIONS
If conflicting information arises from different memories, identify and resolve these contradictions.
- **Identify Conflicts:** Highlight any discrepancies in the data (e.g., "Memory A states X, while Memory B states not X").
- **Resolution Strategy:** Determine which memory is more reliable based on recency, source credibility, or corroboration from other memories.
- **Resolve Conflicts:** Clearly state which information is accepted and which is discarded, along with the rationale.

## STEP 5: COMPLETENESS CHECK
Review your findings against the "Critical Requirements":
- Did I include all specific names (people, places, organizations)?
- Are all numbers and amounts exact?
- Did I convert all relative time references to specific dates?
- Is the answer derived strictly from the provided memories + logical inference?

## STEP 6: FINAL ANSWER
Provide the final, concise answer to the user.
- Write naturally but maintain high information density.
- Provide your final answer after "FINAL ANSWER:". The final answer should be concise yet include all necessary details.

---

**INPUT DATA:**

Memories:
{{ memories }}

Question: {{ question }}
Question Date: {{ question_date }}

**OUTPUT:**
"""


ANSWER_PROMPT_EVERMEMOS_LONGMEMEVAL = """You are an intelligent memory assistant tasked with retrieving accurate information from memories.

# CONTEXT:
You have access to memories from conversations between a user and an AI assistant. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. 
Do not dismiss reasonable inferences as "speculation." 
Your task is to provide the most complete answer supported by the available evidence. 
If the memories do not contain sufficient information to answer the question, explicitly state that.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

Memories:

{{ memories }}

Question: {{ question }}
Question Date: {{ question_date }}

Now, follow the Chain-of-Thought process above to answer the question:

"""



ANSWER_PROMPT = """
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

    {{memories}}

    Question: {{question}}

    Answer:
    """

