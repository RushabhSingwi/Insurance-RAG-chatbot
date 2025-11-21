import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except (ValueError, AttributeError):
            pass  # Already wrapped or not available

# Add parent directory to path to import embedder
sys.path.append(str(Path(__file__).parent.parent))

from embeddings.embedder import EmbeddingGenerator
from embeddings.build_index import FAISSIndexBuilder

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
ENABLE_ANSWER_GENERATION = os.getenv("ENABLE_ANSWER_GENERATION", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


class RAGPipeline:
    """
    RAG (Retrieval Augmented Generation) Pipeline for IRDAI Insurance Circulars.

    This pipeline:
    1. Takes a user query
    2. Generates an embedding for the query
    3. Searches the FAISS index for similar document chunks
    4. Returns relevant context that can be used for answer generation
    """

    def __init__(
        self,
        vector_store_dir: str = VECTOR_STORE_DIR,
        embedding_provider: str = EMBEDDING_PROVIDER,
        embedding_model: str = None,
        enable_answer_generation: bool = ENABLE_ANSWER_GENERATION,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store_dir: Directory containing the FAISS index
            embedding_provider: Provider for embeddings ('sentence-transformers' or 'openai')
            embedding_model: Model name (optional, uses env var if not provided)
            enable_answer_generation: Whether to enable LLM-based answer generation
            llm_provider: LLM provider to use ('groq', 'ollama', 'huggingface')
        """
        # Convert to absolute path if relative
        vector_store_path = Path(vector_store_dir)
        if not vector_store_path.is_absolute():
            # Get the project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            vector_store_path = project_root / vector_store_dir

        self.vector_store_dir = vector_store_path
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.enable_answer_generation = enable_answer_generation
        self.answer_generator = None

        print("Initializing RAG Pipeline...")
        print(f"  Vector store: {vector_store_dir}")
        print(f"  Embedding provider: {embedding_provider}")
        print(f"  Answer generation: {'Enabled' if enable_answer_generation else 'Disabled'}")

        # Initialize embedder
        print("\nLoading embedding model...")
        self.embedder = EmbeddingGenerator(provider=embedding_provider, model=embedding_model)
        self.embedding_dim = self.embedder.get_embedding_dimension()

        # Load FAISS index
        print("\nLoading FAISS index...")
        self.index_builder = FAISSIndexBuilder(embedding_dim=self.embedding_dim)

        index_path = self.vector_store_dir / "faiss_index.bin"
        chunks_path = self.vector_store_dir / "chunks.json"
        metadata_path = self.vector_store_dir / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Please run build_index.py first to create the index."
            )

        self.index_builder.load_index(
            str(index_path),
            str(chunks_path),
            str(metadata_path)
        )

        print(f"\n[OK] RAG Pipeline initialized successfully!")
        print(f"  Total documents in index: {self.index_builder.index.ntotal}")
        print(f"  Embedding dimension: {self.embedding_dim}")

        # Initialize answer generator if enabled
        if self.enable_answer_generation:
            try:
                from llm.answer_generator import AnswerGenerator
                provider = llm_provider or LLM_PROVIDER
                print(f"\nInitializing answer generator with {provider}...")
                self.answer_generator = AnswerGenerator(provider=provider)
                print("[OK] Answer generator ready!")
            except Exception as e:
                print(f"[WARNING] Could not initialize answer generator: {e}")
                print("  Falling back to retrieval-only mode.")
                self.enable_answer_generation = False

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS, boost_sources: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query string
            top_k: Number of top results to return
            boost_sources: Optional list of source files to prioritize in results

        Returns:
            List of dictionaries containing chunks, metadata, and similarity scores
        """
        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query)

        if query_embedding is None:
            print("Error: Failed to generate query embedding")
            return []

        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search index (retrieve more if we need to boost certain sources)
        search_k = top_k * 3 if boost_sources else top_k
        results = self.index_builder.search(query_embedding, k=search_k)

        # If boost_sources is provided, prioritize chunks from those sources
        if boost_sources:
            boosted_results = []
            other_results = []

            for result in results:
                if result['metadata']['source_file'] in boost_sources:
                    # Boost by reducing distance (making it more similar)
                    result['distance'] = result['distance'] * 0.7  # 30% boost
                    boosted_results.append(result)
                else:
                    other_results.append(result)

            # Combine and sort by distance
            all_results = boosted_results + other_results
            all_results.sort(key=lambda x: x['distance'])

            # Return top_k results
            return all_results[:top_k]

        return results

    def format_context(self, results: List[Dict], max_context_length: int = 3000) -> str:
        """
        Format retrieved results into a context string for LLM.

        Args:
            results: List of search results
            max_context_length: Maximum character length for context

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            metadata = result['metadata']
            distance = result['distance']

            # Format the chunk with metadata
            chunk_text = f"[Source: {metadata['source_file']}]\n{chunk}\n"

            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n---\n".join(context_parts)
        return context

    def query(self, question: str, top_k: int = TOP_K_RESULTS, return_context_only: bool = False, conversation_history: Optional[List[Dict]] = None, llm_provider: Optional[str] = None) -> Dict:
        """
        Process a query through the RAG pipeline.

        Args:
            question: User's question
            top_k: Number of relevant documents to retrieve
            return_context_only: If True, only return context without generating answer
            conversation_history: Optional list of previous Q&A for context-aware answers
            llm_provider: Optional LLM provider to use (openai or groq), overrides default

        Returns:
            Dictionary containing:
            - question: Original question
            - context: Retrieved relevant context
            - sources: List of source documents
            - top_k_results: Raw search results
        """
        print(f"\nQuery: {question}")
        print(f"Retrieving top {top_k} relevant documents...")

        # Check if this is a follow-up question by detecting if conversation history has sources
        boost_sources = None
        enhanced_question = question

        if conversation_history and len(conversation_history) > 0:
            # Get sources from the last exchange to boost them in retrieval
            last_entry = conversation_history[-1]
            if last_entry.get('sources'):
                boost_sources = last_entry['sources']
                print(f"  Boosting {len(boost_sources)} previous sources:")
                for src in boost_sources:
                    print(f"    - {src}")

                # Enhance vague follow-up questions with context from previous question
                # This helps retrieval find the right documents
                vague_starters = ['when', 'what', 'where', 'who', 'how', 'why', 'it', 'this', 'that', 'they']

                if len(question.split()) < 10 and any(question.lower().startswith(starter) for starter in vague_starters):
                    # Add context from previous question to make query more specific
                    prev_q = last_entry.get('question', '')
                    if prev_q and len(prev_q.split()) > 3:
                        # Extract key terms from previous question (nouns/important words)
                        key_terms = ' '.join([w for w in prev_q.split() if len(w) > 4])[:100]
                        enhanced_question = f"{question} {key_terms}"
                        print(f"  Enhanced query for better retrieval: {enhanced_question[:80]}...")

        # Retrieve relevant documents
        results = self.retrieve(enhanced_question, top_k=top_k, boost_sources=boost_sources)

        if not results:
            return {
                'question': question,
                'answer': 'I cannot answer this question as no relevant documents were found in the IRDAI circulars database.',
                'context': 'No relevant context found.',
                'sources': [],
                'top_k_results': []
            }

        # Format context
        context = self.format_context(results)

        # Extract unique sources
        sources = list(set([r['metadata']['source_file'] for r in results]))

        print(f"\n Retrieved {len(results)} relevant chunks from {len(sources)} documents")

        # Store actual retrieved sources (not just what LLM cites in answer)
        # This ensures follow-up questions boost the right documents
        actual_retrieved_sources = sources.copy()

        response = {
            'question': question,
            'context': context,
            'sources': sources,
            'top_k_results': results,
            'actual_retrieved_sources': actual_retrieved_sources  # Track what was actually retrieved
        }

        # Generate answer if enabled
        if self.enable_answer_generation and self.answer_generator and not return_context_only:
            print("\nGenerating answer...")
            try:
                # If a different provider is requested, reinitialize the generator
                if llm_provider and llm_provider != self.answer_generator.provider:
                    print(f"Switching LLM provider to: {llm_provider}")
                    from llm.answer_generator import AnswerGenerator
                    temp_generator = AnswerGenerator(provider=llm_provider)
                    answer_result = temp_generator.generate_answer(
                        question=question,
                        context=context,
                        sources=sources,
                        conversation_history=conversation_history
                    )
                else:
                    answer_result = self.answer_generator.generate_answer(
                        question=question,
                        context=context,
                        sources=sources,
                        conversation_history=conversation_history
                    )
                response['answer'] = answer_result.get('answer', 'Failed to generate answer.')
                response['llm_provider'] = answer_result.get('provider', 'unknown')
                response['llm_model'] = answer_result.get('model', 'unknown')
                if 'tokens_used' in answer_result and answer_result['tokens_used']:
                    response['tokens_used'] = answer_result['tokens_used']
                print("Answer generated successfully!")
            except Exception as e:
                print(f"Error generating answer: {e}")
                response['answer'] = f"Retrieved relevant documents but failed to generate answer: {str(e)}"
        else:
            response['answer'] = None

        return response

    def display_results(self, query_result: Dict):
        """
        Display query results in a readable format.

        Args:
            query_result: Result dictionary from query()
        """
        print("\n" + "="*80)
        print("RAG QUERY RESULTS")
        print("="*80)

        print(f"\nQuestion: {query_result['question']}")

        # Display the generated answer if available
        if 'answer' in query_result and query_result['answer']:
            print(f"\n{'='*80}")
            print("ANSWER:")
            print("="*80)
            print(query_result['answer'])
            if 'llm_provider' in query_result:
                print(f"\n[Generated using: {query_result.get('llm_provider', 'unknown')} - {query_result.get('llm_model', 'unknown')}]")
            print("="*80)

        print(f"\nSources ({len(query_result['sources'])} documents):")
        for source in query_result['sources']:
            print(f"  {source}")

        print(f"\nRetrieved Context:")
        print("-"*80)
        print(query_result['context'])
        print("-"*80)

        print(f"\nTop {len(query_result['top_k_results'])} Results with Scores:")
        for i, result in enumerate(query_result['top_k_results'], 1):
            distance = result['distance']
            # Convert L2 distance to similarity score (lower distance = higher similarity)
            similarity = 1 / (1 + distance)
            print(f"\n{i}. [Similarity: {similarity:.4f} | Distance: {distance:.4f}]")
            print(f"   Source: {result['metadata']['source_file']}")
            print(f"   Chunk: {result['chunk'][:200]}...")


def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    print("="*80)
    print("IRDAI Insurance Circulars RAG Pipeline")
    print("="*80 + "\n")

    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the following command first:")
        print("  python src/embeddings/build_index.py")
        return


    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nYou can now ask questions about IRDAI insurance circulars.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_query = input("\nYour question: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting RAG pipeline. Goodbye!")
                break

            if not user_query:
                print("Please enter a valid question.")
                continue

            result = pipeline.query(user_query, top_k=5)
            pipeline.display_results(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")


if __name__ == "__main__":
    main()