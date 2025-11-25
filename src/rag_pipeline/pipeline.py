import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Fix hnswlib compatibility issue
try:
    import hnswlib
    if not hasattr(hnswlib.Index, 'file_handle_count'):
        hnswlib.Index.file_handle_count = 1
except ImportError:
    pass

# Load environment variables (override=True to ensure .env values take precedence)
load_dotenv(override=True)

# Get project root (rag-irdai-chatbot directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Use absolute path to vector DB to avoid issues when running from different directories
CHROMADB_VECTOR_DB = os.getenv("CHROMADB_VECTOR_DB", str(PROJECT_ROOT / "data" / "chromadb"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
ENABLE_ANSWER_GENERATION = os.getenv("ENABLE_ANSWER_GENERATION", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


class RAGPipeline:
    """
    RAG Pipeline for IRDAI Insurance Circulars.
    """

    def __init__(
        self,
        vector_db_path: str = CHROMADB_VECTOR_DB,
        enable_answer_generation: bool = ENABLE_ANSWER_GENERATION,
        llm_provider: Optional[str] = None
    ):
        """Initialize the RAG pipeline."""
        self.vector_db_path = vector_db_path
        self.enable_answer_generation = enable_answer_generation
        self.answer_generator = None

        print("Initializing RAG Pipeline...")
        print(f"  Vector store: {vector_db_path}")
        print(f"  Embedding provider: {EMBEDDING_PROVIDER}")
        print(f"  Embedding model: {EMBEDDING_MODEL}")

        # Load embeddings based on provider
        if EMBEDDING_PROVIDER == "openai":
            embeddings_model = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
        elif EMBEDDING_PROVIDER == "huggingface":
            # Use local HuggingFace embeddings (free, no API key needed)
            embeddings_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")

        # Load vector store
        print("\nLoading Chroma vector store...")
        self.vector_store = Chroma(
            collection_name="irdai_documents",
            embedding_function=embeddings_model,
            persist_directory=vector_db_path
        )

        count = self.vector_store._collection.count()
        print(f"\n[OK] RAG Pipeline initialized!")
        print(f"  Total documents: {count}")

        # Initialize answer generator if enabled
        if self.enable_answer_generation:
            try:
                sys.path.append(str(Path(__file__).parent.parent))
                from llm.answer_generator import AnswerGenerator
                provider = llm_provider or LLM_PROVIDER
                print(f"\nInitializing answer generator with {provider}...")
                self.answer_generator = AnswerGenerator(provider=provider)
                print("[OK] Answer generator ready!")
            except Exception as e:
                print(f"[WARNING] Could not initialize answer generator: {e}")
                self.enable_answer_generation = False

    def query(self, question: str, top_k: int = TOP_K_RESULTS, llm_provider: Optional[str] = None, conversation_history: Optional[List[Dict]] = None, original_question: Optional[str] = None) -> Dict:
        """Process a query through the RAG pipeline.

        Args:
            question: The user's question (may be rephrased for better retrieval)
            top_k: Number of results to retrieve
            llm_provider: Optional LLM provider to use ('openai' or 'groq'). If None, uses default.
            conversation_history: Optional list of previous Q&A pairs for context
            original_question: Original user question before rephrasing (used for LLM prompt)
        """
        # Use original question for display and LLM, rephrased question for retrieval
        retrieval_query = question
        user_question = original_question if original_question else question

        # DEBUG: Show which question will be used
        if os.getenv("DEBUG", "false").lower() == "true":
            print("\n" + "="*80)
            print("DEBUG: QUESTION ROUTING")
            print("="*80)
            print(f"Received question (may be rephrased): {question}")
            print(f"Original question (for LLM): {user_question}")
            print(f"Retrieval query (initial): {retrieval_query}")
            print("="*80)

        # LLM-based query rewriting if conversation history is provided
        if conversation_history and len(conversation_history) > 0 and self.answer_generator:
            try:
                # DEBUG: Show conversation history received
                if os.getenv("DEBUG", "false").lower() == "true":
                    print("\n" + "="*80)
                    print("DEBUG: CONVERSATION HISTORY RECEIVED IN PIPELINE")
                    print("="*80)
                    print(f"Number of exchanges: {len(conversation_history)}")
                    for i, entry in enumerate(conversation_history, 1):
                        print(f"\nExchange {i}:")
                        print(f"  Q: {entry.get('question', '')[:100]}...")
                        print(f"  A: {entry.get('answer', '')[:100]}...")
                        print(f"  Sources: {entry.get('sources', [])}")

                # Format recent conversation context
                history_context = []
                for entry in conversation_history[-2:]:  # Last 2 exchanges
                    q = entry.get('question', '')
                    a = entry.get('answer', '')[:200]  # Truncate answer
                    history_context.append(f"Q: {q}\nA: {a}")

                history_str = "\n\n".join(history_context)

                # Use LLM to rewrite query
                rewrite_prompt = f"Rewrite the following query using prior conversation context so it can be searched standalone. Keep it concise.\n\nHistory:\n{history_str}\n\nQuery: {question}\n\nRewritten query:"

                # DEBUG: Show rewrite prompt
                if os.getenv("DEBUG", "false").lower() == "true":
                    print("\n" + "="*80)
                    print("DEBUG: QUERY REWRITE PROMPT")
                    print("="*80)
                    print(rewrite_prompt)
                    print("="*80)

                if self.answer_generator.provider == "openai":
                    from langchain_core.messages import HumanMessage
                    response = self.answer_generator.client.invoke(
                        [HumanMessage(content=rewrite_prompt)],
                        temperature=0.3,
                        max_tokens=50
                    )
                    rewritten_query = response.content.strip()
                elif self.answer_generator.provider == "groq":
                    response = self.answer_generator.client.chat.completions.create(
                        model=self.answer_generator.model,
                        messages=[{"role": "user", "content": rewrite_prompt}],
                        temperature=0.3,
                        max_tokens=50
                    )
                    rewritten_query = response.choices[0].message.content.strip()
                else:
                    rewritten_query = None

                # Use rewritten query if successful
                if rewritten_query and len(rewritten_query) > 5:
                    retrieval_query = rewritten_query
                    print(f"  Query rewritten: {question} → {rewritten_query}")
            except Exception as e:
                print(f"  Query rewrite failed: {e}, using original")

        print(f"\nQuery: {user_question}")
        if retrieval_query != user_question:
            print(f"  (Retrieval query: {retrieval_query})")
        print(f"Retrieving top {top_k} relevant documents...")

        results = self.vector_store.similarity_search_with_score(retrieval_query, k=top_k)

        formatted_results = []
        for doc, distance in results:
            formatted_results.append({
                'chunk': doc.page_content,
                'metadata': doc.metadata,
                'distance': float(distance)
            })

        if not formatted_results:
            return {
                'question': user_question,  # Use original question, not rephrased
                'answer': 'No relevant documents found.',
                'context': 'No relevant context found.',
                'sources': [],
                'results': []
            }

        # Format context
        context_parts = []
        for result in formatted_results:
            chunk = result['chunk']
            metadata = result['metadata']
            context_parts.append(f"[Source: {metadata['source_file']}]\n{chunk}\n")

        context = "\n---\n".join(context_parts)
        sources = list(set([r['metadata']['source_file'] for r in formatted_results]))

        print(f"\nRetrieved {len(formatted_results)} chunks from {len(sources)} documents")

        # Debug: Show distance scores for top results
        if formatted_results and os.getenv("DEBUG", "false").lower() == "true":
            print("\nTop retrieval scores:")
            for i, result in enumerate(formatted_results[:3], 1):
                print(f"  {i}. {result['metadata']['source_file'][:50]}: distance={result['distance']:.4f}")

        response = {
            'question': user_question,  # Return original question, not rephrased
            'context': context,
            'sources': sources,
            'results': formatted_results,
        }

        # Generate answer if enabled
        if self.enable_answer_generation and self.answer_generator:
            print("\nGenerating answer...")
            try:
                # Use dynamic LLM provider if specified, otherwise use default generator
                if llm_provider and llm_provider != self.answer_generator.provider:
                    # Create a new generator with the requested provider
                    from llm.answer_generator import AnswerGenerator
                    temp_generator = AnswerGenerator(provider=llm_provider)
                    answer_result = temp_generator.generate_answer(
                        question=user_question,  # Use original question for LLM prompt
                        context=context,
                        conversation_history=conversation_history
                    )
                else:
                    # Use the default generator
                    answer_result = self.answer_generator.generate_answer(
                        question=user_question,  # Use original question for LLM prompt
                        context=context,
                        conversation_history=conversation_history
                    )
                response['answer'] = answer_result.get('answer', 'Failed to generate answer.')
                response['llm_provider'] = answer_result.get('provider', 'unknown')
                response['llm_model'] = answer_result.get('model', 'unknown')

                # Include token usage if available (OpenAI only)
                if answer_result.get('tokens_used'):
                    response['tokens_used'] = answer_result.get('tokens_used')
                    response['prompt_tokens'] = answer_result.get('prompt_tokens')
                    response['completion_tokens'] = answer_result.get('completion_tokens')

                print("Answer generated successfully!")
            except Exception as e:
                # Handle encoding issues in error messages
                error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
                print(f"Error generating answer: {error_msg}")
                response['answer'] = f"Retrieved documents but failed to generate answer: {error_msg}"
        else:
            response['answer'] = None

        return response


def main():
    """Demo the RAG pipeline."""
    print("="*80)
    print("IRDAI Insurance Circulars RAG Pipeline")
    print("="*80 + "\n")

    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease run: python src/embeddings/build_index.py")
        return

    # Interactive mode
    print("\n" + "="*80)
    print("Ask questions about IRDAI insurance circulars.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_query = input("\nYour question: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_query:
                continue

            result = pipeline.query(user_query, top_k=5)

            # Display answer
            if result.get('answer'):
                print(f"\n{'='*80}")
                print("ANSWER:")
                print("="*80)
                print(result['answer'])
                print("="*80)

            # Display sources
            print(f"\nSources ({len(result['sources'])} documents):")
            for source in result['sources']:
                print(f"  • {source}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
