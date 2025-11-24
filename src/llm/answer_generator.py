"""
Answer Generation Module using LLM APIs

Supports multiple LLM providers:
1. OpenAI API
2. Groq API (Free tier - Fast inference)
3. Ollama (Local - Completely free)
4. HuggingFace Inference API (Free tier)
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # openai, groq, ollama, or huggingface
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class AnswerGenerator:
    """
    Generate answers from retrieved context using LLMs.
    """

    def __init__(self, provider: str = LLM_PROVIDER):
        """
        Initialize the answer generator.

        Args:
            provider: LLM provider to use ('openai', 'groq', 'ollama', or 'huggingface')
        """
        self.provider = provider.lower()
        self.client = None

        print(f"Initializing Answer Generator with provider: {self.provider}")

        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "groq":
            self._init_groq()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_openai(self):
        """Initialize OpenAI client using LangChain."""
        try:
            from langchain_openai import ChatOpenAI

            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Get an API key from: https://platform.openai.com/api-keys"
                )

            self.client = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.1
            )
            self.model = OPENAI_MODEL
            print(f"OpenAI ChatOpenAI initialized with model: {self.model}")

        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")

    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq

            if not GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY not found in environment variables. "
                    "Get a free API key from: https://console.groq.com"
                )

            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = GROQ_MODEL
            print(f"Groq client initialized with model: {self.model}")

        except ImportError:
            raise ImportError("Please install groq: pip install groq")

    def create_prompt(self, question: str, context: str, sources: List[str], conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Create a prompt for the LLM with few-shot examples and conversation context.

        Args:
            question: User's question
            context: Retrieved context from RAG
            sources: List of source documents
            conversation_history: Optional list of previous Q&A pairs for follow-up questions

        Returns:
            Formatted prompt string
        """
        # Build conversation context if provided
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
            conversation_context += "(Use these sources to answer follow-up questions about dates, details, or clarifications)\n\n"
            for i, entry in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
                conversation_context += f"Q{i}: {entry.get('question', '')}\n"
                conversation_context += f"A{i}: {entry.get('answer', '')[:300]}...\n"
                if entry.get('sources'):
                    conversation_context += f"Sources used: {', '.join(entry['sources'][:3])}\n"
                conversation_context += "\n"
            conversation_context += "IMPORTANT: If the current question asks about dates, decisions, or additional details related to the above conversation, retrieve information from the SAME sources mentioned above.\n"

        prompt = f"""You are an expert AI assistant for IRDAI (Insurance Regulatory and Development Authority of India) insurance regulations. Your role is to provide accurate, helpful answers based EXCLUSIVELY on the provided context documents.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL GROUNDING RULES - NEVER VIOLATE THESE:
═══════════════════════════════════════════════════════════════════════════════

1. ✓ ONLY use information from the CONTEXT section below
2. ✗ NEVER use your training data, general knowledge, or make assumptions
3. ✓ If the answer requires synthesis from MULTIPLE parts of the context, do so
4. ✓ If information is IMPLIED or can be REASONABLY INFERRED from the context, state it clearly
5. ✗ If the answer is NOT in the context (even partially), respond: "I cannot answer this based on the available documents."

═══════════════════════════════════════════════════════════════════════════════
HOW TO EXTRACT INFORMATION FROM CONTEXT:
═══════════════════════════════════════════════════════════════════════════════

STEP 1: SCAN ALL CONTEXT THOROUGHLY
- Read through ALL provided context chunks, not just the first one
- Key information may be split across multiple chunks from the same document
- Look for: numbers, dates, percentages, procedures, requirements, definitions

STEP 2: IDENTIFY RELEVANT INFORMATION
- Extract EXACT values: numbers, percentages, dates, thresholds
- Note document references: "Ref: IRDAI/..." and dates at document beginnings
- Capture procedural details: requirements, steps, conditions, obligations
- Synthesize if information appears in multiple chunks

STEP 3: STRUCTURE YOUR ANSWER
- Start with DIRECT ANSWER (include specific numbers/thresholds/dates)
- Add supporting details if present in context
- Cite source document(s)
- Use clear formatting (bold for key facts)

STEP 4: DATE EXTRACTION PROTOCOL
- Dates often appear at document START: "Date: [DD Month YYYY]" or within "Ref:" line
- Sometimes dates are in format: "DD.MM.YYYY" or "DD/MM/YYYY"
- Look for phrases like "dated", "passed on", "issued on"
- If date is partially readable (e.g., "00 January" due to OCR), infer from reference number

═══════════════════════════════════════════════════════════════════════════════
SPECIAL CASES:
═══════════════════════════════════════════════════════════════════════════════

DEFINITIONS/CONCEPTS (e.g., "What is X?"):
- Provide complete definition synthesized from all relevant context
- Include purpose, mechanism, scope, and key features
- Example: If asked "What is Bima-ASBA?", combine all context about it into coherent explanation

REQUIREMENTS/PROCEDURES (e.g., "What are the requirements for X?"):
- List all requirements/steps found in context
- Maintain numbering if present
- Clarify if requirements are mandatory vs optional

COMPARISONS/MULTIPLE ITEMS:
- Synthesize information across context chunks
- Create structured comparison if helpful
- Ensure all facts come from provided context

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES OF CORRECT ANSWERING:
═══════════════════════════════════════════════════════════════════════════════

Example 1 - EXTRACTING SPECIFIC VALUES:
CONTEXT:
[Source: Review of revision in premium rates under health insurance policies for senior citizens]
Ref: IRDAI/HLT/CIR/MISC/27/1/2025
Date: 30th January, 2025
The IRDAI hereby directs:
a) Insurers shall not revise premium for senior citizens by more than 10% per annum.
b) For increases above 10%, prior consultation with IRDAI is required.

QUESTION: What is the threshold for revising insurance premium for senior citizens?

✓ CORRECT ANSWER:
The threshold for revising health insurance premium for senior citizens is **10% per annum**. Insurers cannot increase premiums by more than 10% per year without prior consultation with IRDAI.

**Source:** Review of revision in premium rates under health insurance policies for senior citizens (Ref: IRDAI/HLT/CIR/MISC/27/1/2025)

---

Example 2 - EXTRACTING DATES:
CONTEXT:
[Source: Review of revision in premium rates under health insurance policies for senior citizens]
Ref: IRDAI/HLT/CIR/MISC/27/1/2025
Date: 30th January, 2025
The IRDAI hereby directs all general and health insurers...

QUESTION: When was the decision about senior citizen premium revision passed?

✓ CORRECT ANSWER:
The decision was passed on **30th January, 2025** (Reference: IRDAI/HLT/CIR/MISC/27/1/2025).

**Source:** Review of revision in premium rates under health insurance policies for senior citizens

---

Example 3 - SYNTHESIZING DEFINITION FROM MULTIPLE CHUNKS:
CONTEXT (multiple chunks from same document):
[Chunk 1] Bima-ASBA stands for Bima Applications Supported by Blocked Amount. It is a facility using UPI One Time Mandate (OTM) for premium payment.
[Chunk 2] Under Bima-ASBA, insurers can block premium amount in prospect's bank account via UPI. Amount is debited only after proposal acceptance.
[Chunk 3] If proposal rejected, blocked amount is automatically released. Mandate valid for maximum 14 days.

QUESTION: What is Bima-ASBA and how does it work?

✓ CORRECT ANSWER:
**Bima-ASBA** (Bima Applications Supported by Blocked Amount) is a premium payment facility using UPI One Time Mandate that allows insurers to block funds in a prospect's bank account for insurance premium.

**How it works:**
1. Insurer blocks the premium amount in the prospect's account via UPI
2. Funds remain blocked (not debited) until underwriting decision is made
3. If proposal is accepted: Amount is debited and policy is issued
4. If proposal is rejected: Blocked amount is automatically released
5. Mandate is valid for maximum 14 days

**Source:** One-time Mandate for blocking the amount towards premium through UPI for issuance of life and health insurance policies- Bima-ASBA

---

Example 4 - LISTING REQUIREMENTS:
CONTEXT:
[Source: Master Circular on Rural Obligations]
Every insurer shall:
a) Cover minimum 15% of lives in allocated Gram Panchayats
b) Submit quarterly reports to IRDAI
c) Obtain certificates from Gram Sachiv
d) Coordinate with respective Councils

QUESTION: What are the rural sector obligations for insurers?

✓ CORRECT ANSWER:
Insurers must fulfill the following rural sector obligations:

1. **Coverage:** Cover minimum 15% of lives in allocated Gram Panchayats
2. **Reporting:** Submit quarterly reports to IRDAI
3. **Certification:** Obtain certificates from Gram Sachiv
4. **Coordination:** Coordinate with respective Councils

**Source:** Master Circular on Rural Obligations

---

═══════════════════════════════════════════════════════════════════════════════
NOW ANSWER THE USER'S QUESTION
═══════════════════════════════════════════════════════════════════════════════
{conversation_context}

CONTEXT (Read ALL chunks carefully - information may be split across multiple chunks):
{context}

SOURCE DOCUMENTS AVAILABLE:
{', '.join(sources)}

USER QUESTION:
{question}

INSTRUCTIONS FOR YOUR ANSWER:
1. Scan ALL context chunks above thoroughly
2. Extract relevant information (numbers, dates, procedures, definitions)
3. Synthesize if information is split across chunks
4. Format answer with key facts in bold
5. Cite source document(s)
6. If answer not found in context, respond: "I cannot answer this based on the available documents."

YOUR ANSWER (with citations):"""

        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        sources: List[str],
        temperature: float = 0.1,
        max_tokens: int = 500,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        Generate an answer using the LLM.

        Args:
            question: User's question
            context: Retrieved context from RAG
            sources: List of source documents
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            conversation_history: Optional list of previous Q&A pairs for context

        Returns:
            Dictionary with 'answer' and 'provider' keys
        """
        prompt = self.create_prompt(question, context, sources, conversation_history)

        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, temperature, max_tokens)
            elif self.provider == "groq":
                return self._generate_groq(prompt, temperature, max_tokens)
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "provider": self.provider,
                "error": str(e)
            }

    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, str]:
        """Generate answer using LangChain ChatOpenAI with retry logic for rate limits."""
        import time
        from langchain_core.messages import SystemMessage, HumanMessage

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Create messages for LangChain
                messages = [
                    SystemMessage(content="You are an expert IRDAI insurance regulations assistant. You must answer ONLY using the provided context. Synthesize information from multiple chunks when needed. Extract exact values, dates, and procedures. Do NOT use your training data. If the answer is not in the context, say 'I cannot answer this based on the available documents.'"),
                    HumanMessage(content=prompt)
                ]

                # Invoke the LLM with callbacks to track token usage
                response = self.client.invoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Extract token usage from response metadata
                tokens_used = None
                prompt_tokens = None
                completion_tokens = None

                if hasattr(response, 'response_metadata') and response.response_metadata:
                    token_usage = response.response_metadata.get('token_usage', {})
                    tokens_used = token_usage.get('total_tokens')
                    prompt_tokens = token_usage.get('prompt_tokens')
                    completion_tokens = token_usage.get('completion_tokens')

                # Log token usage
                if tokens_used:
                    print(f"\n[OpenAI Token Usage]")
                    print(f"  Prompt tokens: {prompt_tokens}")
                    print(f"  Completion tokens: {completion_tokens}")
                    print(f"  Total tokens: {tokens_used}")
                    print(f"  Model: {self.model}")

                return {
                    "answer": response.content.strip(),
                    "provider": "openai",
                    "model": self.model,
                    "tokens_used": tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }

            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error (429)
                if "429" in error_msg and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                    print(f"\n[WARN] Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue  # Retry

                # If not a rate limit, or we've exhausted retries, show detailed error
                print(f"[ERROR] OpenAI API call failed: {e}")

                # Check if it's a rate limit or quota error
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    print("\n[WARN] Rate limit or quota issue detected!")
                    print("Possible causes:")
                    print("  1. Request rate limit (requests per minute)")
                    print("  2. Token rate limit (tokens per minute)")
                    print("  3. Insufficient quota/credits")
                    print("\nSolutions:")
                    print("  - Wait 60 seconds and try again")
                    print("  - Check your usage: https://platform.openai.com/usage")
                    print("  - Upgrade tier: https://platform.openai.com/settings/organization/billing")
                    print("  - Or switch to Groq (free): Set LLM_PROVIDER=groq in .env")

                import traceback
                traceback.print_exc()
                raise

    def _generate_groq(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, str]:
        """Generate answer using Groq."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert IRDAI insurance regulations assistant. You must answer ONLY using the provided context. Synthesize information from multiple chunks when needed. Extract exact values, dates, and procedures. Do NOT use your training data. If the answer is not in the context, say 'I cannot answer this based on the available documents.'"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return {
                "answer": response.choices[0].message.content.strip(),
                "provider": "groq",
                "model": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# Example usage
if __name__ == "__main__":
    # Test the answer generator
    generator = AnswerGenerator(provider="groq")

    question = "What is the threshold for revising insurance premium for senior citizens?"
    context = """
    [Source: IRDAI Circular on Health Insurance Premium]
    The Insurance Regulatory and Development Authority of India (IRDAI) has decided that insurance companies
    can revise health insurance premiums for senior citizens by up to 10% without requiring prior approval.
    This decision was made on 30th January 2025 to provide flexibility to insurers while protecting consumers.
    For revisions above 10%, prior approval from IRDAI is required.
    """
    sources = ["IRDAI Circular on Health Insurance Premium"]

    result = generator.generate_answer(question, context, sources)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nProvider: {result['provider']}")
    print(f"Model: {result['model']}")
