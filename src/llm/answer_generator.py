"""
Answer Generation Module

Generates answers from retrieved context using LLM APIs.
Supports multiple providers: OpenAI and Groq.
Includes conversation history support and retry logic for rate limiting.
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from groq import Groq

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config
from utils.debug_utils import is_debug_basic, is_debug_verbose, print_debug_header, print_debug_footer

# Load configuration
config = get_config()

# Constants
MAX_CONVERSATION_HISTORY = 3  # Keep last N exchanges
MAX_ANSWER_LENGTH = 200  # Characters for conversation context
RATE_LIMIT_RETRY_COUNT = 3
RATE_LIMIT_BASE_DELAY = 1  # seconds

class AnswerGenerator:
    """
    Generate answers from retrieved context using LLMs.

    Supports OpenAI and Groq providers with conversation history,
    token tracking, and rate limit handling.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the answer generator.

        Args:
            provider: LLM provider to use ('openai' or 'groq'). If None, uses default from config.
        """
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.client = None

        print(f"Initializing Answer Generator with provider: {self.provider}")

        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "groq":
            self._init_groq()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Must be 'openai' or 'groq'.")

        # Store system message as instance attribute
        self.system_message = f"""You are an AI assistant specialized in Indian insurance regulations from IRDAI (Insurance Regulatory and Development Authority of India).

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

---"""

    def _init_openai(self) -> None:
        """Initialize OpenAI client using LangChain."""
        try:
            if not config.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Get an API key from: https://platform.openai.com/api-keys"
                )

            self.client = ChatOpenAI(
                model=config.OPENAI_MODEL,
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            self.model = config.OPENAI_MODEL
            print(f"OpenAI ChatOpenAI initialized with model: {self.model}")

        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")

    def _init_groq(self) -> None:
        """Initialize Groq client."""
        try:
            if not config.GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY not found in environment variables. "
                    "Get a free API key from: https://console.groq.com"
                )

            self.client = Groq(api_key=config.GROQ_API_KEY)
            self.model = config.GROQ_MODEL
            print(f"Groq client initialized with model: {self.model}")

        except ImportError:
            raise ImportError("Please install groq: pip install groq")

    def create_prompt(self, question: str, context: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Create a prompt for the LLM with conversation context.

        Args:
            question: User's question
            context: Retrieved context from RAG
            conversation_history: Optional list of previous Q&A pairs for follow-up questions

        Returns:
            Formatted prompt string
        """
        # Build conversation context if provided
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
            for entry in conversation_history[-MAX_CONVERSATION_HISTORY:]:
                conversation_context += f"User: {entry.get('question', '')}\n"
                answer_preview = entry.get('answer', '')[:MAX_ANSWER_LENGTH]
                conversation_context += f"Assistant: {answer_preview}...\n\n"

        prompt = f"""CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_context}

USER QUESTION:
{question}

NOW ANSWER BASED ON THE CONTEXT."""

        # DEBUG: Show complete prompt (VERBOSE only)
        if is_debug_verbose():
            print_debug_header("FULL LLM PROMPT", level=2)
            print("SYSTEM MESSAGE (first 500 chars):")
            print(self.system_message[:500] + "...")
            print("\nUSER PROMPT:")
            print(prompt)
            print_debug_footer(level=2)

        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        Generate an answer using the LLM.

        Args:
            question: User's question
            context: Retrieved context from RAG
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            conversation_history: Optional list of previous Q&A pairs for context

        Returns:
            Dictionary with 'answer' and 'provider' keys
        """
        prompt = self.create_prompt(question, context, conversation_history)

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
        """
        Generate answer using LangChain ChatOpenAI with retry logic for rate limits.

        Args:
            prompt: User prompt with context
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with answer, provider, model, and token usage
        """
        for attempt in range(RATE_LIMIT_RETRY_COUNT):
            try:
                # Create messages for LangChain
                messages = [
                    SystemMessage(content=self.system_message),
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

                answer_text = response.content.strip()

                # DEBUG: Basic level - Token usage summary
                if is_debug_basic():
                    print(f"\n[LLM Token Usage] Provider: OpenAI | Model: {self.model}")
                    print(f"  Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {tokens_used}")

                # DEBUG: Verbose level - Full response details
                if is_debug_verbose():
                    print_debug_header("LLM RESPONSE (OpenAI)", level=2)
                    print(f"Model: {self.model}")
                    print(f"Total tokens: {tokens_used} (prompt: {prompt_tokens}, completion: {completion_tokens})")
                    print(f"\nAnswer (first 300 chars):")
                    print(answer_text[:300] + "...")
                    print(f"\nFull answer length: {len(answer_text)} chars")
                    print_debug_footer(level=2)

                return {
                    "answer": answer_text,
                    "provider": "openai",
                    "model": self.model,
                    "tokens_used": tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }

            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error (429)
                if "429" in error_msg and attempt < RATE_LIMIT_RETRY_COUNT - 1:
                    delay = RATE_LIMIT_BASE_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"\n[WARN] Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{RATE_LIMIT_RETRY_COUNT})")
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
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer_text = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else None
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else None

            # DEBUG: Basic level - Token usage summary
            if is_debug_basic():
                print(f"\n[LLM Token Usage] Provider: Groq | Model: {self.model}")
                print(f"  Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {tokens}")

            # DEBUG: Verbose level - Full response details
            if is_debug_verbose():
                print_debug_header("LLM RESPONSE (Groq)", level=2)
                print(f"Model: {self.model}")
                print(f"Total tokens: {tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
                print(f"\nAnswer (first 300 chars):")
                print(answer_text[:300] + "...")
                print(f"\nFull answer length: {len(answer_text)} chars")
                print_debug_footer(level=2)

            return {
                "answer": answer_text,
                "provider": "groq",
                "model": self.model,
                "tokens_used": tokens
            }
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}")
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

    result = generator.generate_answer(question, context)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nProvider: {result['provider']}")
    print(f"Model: {result['model']}")
