"""
STEP 6: Prompt Engineering

System and user prompt templates for citation-grounded medical answer generation.

Model: Groq LLaMA-3 70B
Behavior: Strict evidence-grounded assistant (no hallucinations, mandatory citations)

Templates are LOCKED and must not be modified without careful consideration.
"""

from typing import List, Dict, Any


# SYSTEM PROMPT (LOCK THIS — DO NOT MODIFY)
SYSTEM_PROMPT = """You are a medical information assistant.

You MUST:
- Answer ONLY using the provided context.
- NOT use prior knowledge.
- NOT add new medical facts.
- Cite every factual statement using the provided chunk IDs.
- Refuse diagnosis, prescription, or treatment recommendation requests.

If the answer is not present in the provided context, say:
"I don't have enough information in the provided sources."

Always include the following disclaimer at the end:
"This information is for educational purposes only and is not medical advice." """


# Mandatory disclaimer (must appear at end of every response)
MANDATORY_DISCLAIMER = (
    "This information is for educational purposes only and is not medical advice. "
    "Always consult a qualified healthcare professional for medical concerns."
)


def format_context_chunks(documents: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into context section for prompt.
    
    Each chunk is presented with its ID for citation.
    
    Args:
        documents: List of retrieved documents with id, text, score, metadata
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for doc in documents:
        chunk_id = doc["id"]
        text = doc["text"]
        
        # Format: [CHUNK ID: ...]\n<text>\n
        context_parts.append(f"[CHUNK ID: {chunk_id}]\n{text}")
    
    return "\n\n".join(context_parts)


def build_user_prompt(query: str, documents: List[Dict[str, Any]]) -> str:
    """
    Build user prompt following strict format.
    
    Structure:
    Context:
    [CHUNK ID: ...]
    <text>
    
    Question:
    <query>
    
    Instructions:
    - Answer in clear, concise paragraphs.
    - Use inline citations like (CHUNK_ID).
    - Do not speculate.
    - Do not add information not present in the context.
    
    Args:
        query: User question
        documents: Retrieved context documents
    
    Returns:
        Complete user prompt string
    """
    context = format_context_chunks(documents)
    
    prompt = f"""Context:

{context}

Question:
{query}

Instructions:
- Answer in clear, concise paragraphs.
- IMPORTANT: Cite sources using EXACTLY this format: (CHUNK_ID) with ONE ID per parenthesis.
- Example: "Diabetes affects blood sugar (DOC_001). It can cause complications (DOC_002)."
- NEVER group multiple IDs like (DOC_001, DOC_002) - use separate citations.
- Do not speculate or add information not in the context.
- End your answer with this exact disclaimer: {MANDATORY_DISCLAIMER}"""
    
    return prompt


def get_system_prompt() -> str:
    """
    Get the system prompt.
    
    Returns:
        System prompt string (locked)
    """
    return SYSTEM_PROMPT


def get_mandatory_disclaimer() -> str:
    """
    Get the mandatory disclaimer text.
    
    Returns:
        Disclaimer string
    """
    return MANDATORY_DISCLAIMER


# Example usage for testing
if __name__ == "__main__":
    # Sample documents
    sample_docs = [
        {
            "id": "NCI_DIABETES_SYM_01",
            "text": "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
            "score": 0.85,
            "metadata": {
                "topic": "Symptoms of Diabetes",
                "source": "National Cancer Institute",
                "source_type": "nih"
            }
        },
        {
            "id": "CDC_DIABETES_RISK_02",
            "text": "Risk factors for type 2 diabetes include being overweight, being age 45 or older, having a parent or sibling with type 2 diabetes, being physically active less than 3 times a week, and having had gestational diabetes.",
            "score": 0.78,
            "metadata": {
                "topic": "Risk Factors",
                "source": "CDC",
                "source_type": "public_health"
            }
        }
    ]
    
    sample_query = "What are the symptoms and risk factors of type 2 diabetes?"
    
    print("=" * 70)
    print("PROMPT TEMPLATES - EXAMPLE")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("SYSTEM PROMPT:")
    print("=" * 70)
    print(get_system_prompt())
    
    print("\n" + "=" * 70)
    print("USER PROMPT:")
    print("=" * 70)
    user_prompt = build_user_prompt(sample_query, sample_docs)
    print(user_prompt)
    
    print("\n" + "=" * 70)
    print("MANDATORY DISCLAIMER:")
    print("=" * 70)
    print(get_mandatory_disclaimer())
