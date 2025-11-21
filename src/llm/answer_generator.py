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
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Get an API key from: https://platform.openai.com/api-keys"
                )

            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
            print(f"OpenAI client initialized with model: {self.model}")

        except ImportError:
            raise ImportError("Please install openai: pip install openai")

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

        prompt = f"""You are an AI assistant specialized in Indian insurance regulations from IRDAI (Insurance Regulatory and Development Authority of India).

Your task is to answer questions based ONLY on the provided context from IRDAI circulars. Follow these rules:

1. READ the context CAREFULLY and THOROUGHLY before answering
2. If the question asks for specific numbers, percentages, thresholds, or limits, LOOK for them explicitly in the context
3. Provide a direct, concise answer to the question with the EXACT numbers/percentages if mentioned
4. Use ONLY information from the provided context
5. Include specific citations from the source documents
6. If the context doesn't contain the answer, say "I cannot answer this based on the available documents."
7. Format your answer clearly: main answer first (with numbers/thresholds), then supporting details
8. Always cite the source document name when providing information
9. For follow-up questions, refer to the PREVIOUS CONVERSATION context and use the SAME sources if the question relates to previously discussed topics
10. If asked about dates of decisions/circulars, look for dates at the BEGINNING of the document (common format: "Ref: IRDAI/..." followed by date like "30 January, 2025")

HERE ARE EXAMPLES OF HOW TO ANSWER:

Example 1 - Specific Threshold Question:
CONTEXT:
[Source: Review of revision in premium rates under health insurance policies for senior citizens]
Ref: IRDAI/HLT/CIR/MISC/27/1/2025
30th January, 2025

The IRDAI hereby directs all general and health insurers to take the following steps:
a) The insurers shall not revise the premium for senior citizens by more than 10% per annum.
b) If the increase proposed in the premium for senior citizens is more than 10% per annum, insurers shall undertake prior consultation with the IRDAI.

QUESTION: What is the threshold for revising insurance premium for senior citizens in India for health insurance?

CORRECT ANSWER:
The threshold for revising health insurance premium for senior citizens is **10% per annum**. Insurers cannot revise the premium by more than 10% per year without prior consultation with IRDAI.

**Source:** Review of revision in premium rates under health insurance policies for senior citizens (Ref: IRDAI/HLT/CIR/MISC/27/1/2025)

---

Example 2 - Follow-up Question About Date:
PREVIOUS CONVERSATION:
Q1: What is the threshold for revising insurance premium for senior citizens?
A1: The threshold is 10% per annum.
Sources used: Review of revision in premium rates under health insurance policies for senior citizens

IMPORTANT: The current question is about the date. Look for the date in the CONTEXT from the source mentioned in the previous conversation: "Review of revision in premium rates under health insurance policies for senior citizens"

CONTEXT:
[Source: Review of revision in premium rates under health insurance policies for senior citizens]
Ref: IRDAI/HLT/CIR/MISC/27/1/2025
30th January, 2025

The IRDAI hereby directs all general and health insurers...

[Source: Amendment to Circular on Procedure]
Ref: IRDAI/IID/CIR/MISC/175/10/2023
9th October, 2023

QUESTION: When was this decision passed?

CORRECT ANSWER:
This decision was passed on **30th January, 2025**.

**Source:** Review of revision in premium rates under health insurance policies for senior citizens (Ref: IRDAI/HLT/CIR/MISC/27/1/2025, dated 30th January, 2025)

**Explanation:** I found the date in the same document that was referenced in the previous answer about the 10% threshold for senior citizens.

---

Example 3 - Follow-up About Additional Details:
PREVIOUS CONVERSATION:
Q: What is the threshold for revising insurance premium for senior citizens?
A: The threshold is 10% per annum.

CONTEXT:
[Source: Review of revision in premium rates under health insurance policies for senior citizens]
a) The insurers shall not revise the premium for senior citizens by more than 10% per annum.
b) If the increase proposed in the premium for senior citizens is more than 10% per annum, insurers shall undertake prior consultation with the IRDAI.
c) In case of withdrawal of individual health insurance products offered to senior citizens, insurers shall undertake prior consultation with the IRDAI.

QUESTION: What happens if they want to increase more than this?

CORRECT ANSWER:
If insurers want to increase the premium for senior citizens by more than 10% per annum, they must undertake **prior consultation with IRDAI** before implementing the increase.

**Source:** Review of revision in premium rates under health insurance policies for senior citizens

---

NOW ANSWER THE USER'S QUESTION:{conversation_context}

CONTEXT:
{context}

SOURCE DOCUMENTS:
{', '.join(sources)}

QUESTION:
{question}

ANSWER (with citations):"""

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
        """Generate answer using OpenAI with retry logic for rate limits."""
        import time

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert on IRDAI insurance regulations. Provide accurate, concise answers with proper citations."
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
                    "provider": "openai",
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
                }

            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error (429)
                if "429" in error_msg and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                    print(f"\n⚠️  Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue  # Retry

                # If not a rate limit, or we've exhausted retries, show detailed error
                print(f"[ERROR] OpenAI API call failed: {e}")

                # Check if it's a rate limit or quota error
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    print("\n⚠️  Rate limit or quota issue detected!")
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
                        "content": "You are an expert on IRDAI insurance regulations. Provide accurate, concise answers with proper citations."
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
