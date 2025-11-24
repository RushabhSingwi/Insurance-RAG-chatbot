"""
Performance Benchmarking Script
Measures actual query time, embedding time, and storage metrics
"""

import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()

def benchmark_embeddings():
    """Measure embedding generation time."""
    from src.embeddings.embedder import EmbeddingGenerator

    print("\n" + "="*80)
    print("BENCHMARK: Embedding Generation Time")
    print("="*80)

    embedder = EmbeddingGenerator()

    # Test single query embedding
    test_query = "What are the requirements for cyber security incident reporting?"

    print(f"\nTest Query: '{test_query}'")
    print(f"Provider: {os.getenv('EMBEDDING_PROVIDER', 'sentence-transformers')}")
    print(f"Model: {os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}")

    # Warm up (first call may be slower)
    _ = embedder.embed_text("warmup")

    # Measure 10 queries
    times = []
    for i in range(10):
        start = time.time()
        embedding = embedder.embed_text(test_query)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"  Query {i+1}: {elapsed:.2f}ms")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n📊 Results:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms")
    print(f"  Max: {max_time:.2f}ms")
    print(f"  Embedding dimension: {len(embedding)}")

    return avg_time


def benchmark_chromadb_search():
    """Measure ChromaDB search time."""
    import chromadb
    import numpy as np
    from src.embeddings.build_index import ChromaDBIndexBuilder

    print("\n" + "="*80)
    print("BENCHMARK: ChromaDB Search Time")
    print("="*80)

    vector_store_dir = Path(os.getenv("VECTOR_STORE_DIR", "data/vector_store"))
    persist_dir = vector_store_dir / "chromadb"

    # Check if ChromaDB index exists
    if not persist_dir.exists():
        print("❌ ChromaDB index not found. Run: python src/embeddings/build_index.py")
        return None

    # Load ChromaDB collection
    from src.embeddings.embedder import EmbeddingGenerator
    embedder = EmbeddingGenerator()
    embedding_dim = embedder.get_embedding_dimension()
    
    builder = ChromaDBIndexBuilder(embedding_dim=embedding_dim, persist_directory=str(persist_dir))
    builder.load_index()
    collection = builder._get_or_create_collection()

    print(f"\nIndex Info:")
    print(f"  Total vectors: {collection.count()}")
    print(f"  Collection name: {builder.collection_name}")

    # Create a random query vector
    query_vector = np.random.rand(embedding_dim).astype('float32')

    # Measure 100 searches
    times = []
    for i in range(100):
        start = time.time()
        results = builder.search(query_vector, k=10)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n📊 Results (100 searches, top-10):")
    print(f"  Average: {avg_time:.4f}ms")
    print(f"  Min: {min_time:.4f}ms")
    print(f"  Max: {max_time:.4f}ms")

    return avg_time


def benchmark_full_query():
    """Measure end-to-end query time."""
    from src.rag_pipeline.pipeline import RAGPipeline

    print("\n" + "="*80)
    print("BENCHMARK: End-to-End Query Time")
    print("="*80)

    pipeline = RAGPipeline()

    test_queries = [
        "What are the requirements for cyber security incident reporting?",
        "What is the procedure for motor insurance claims?",
        "What are the guidelines for health insurance policies?",
    ]

    print(f"\nTesting {len(test_queries)} queries...")

    times = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: '{query[:60]}...'")

        start = time.time()
        result = pipeline.query(query)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"    Time: {elapsed:.2f}s")
        if result.get('answer'):
            print(f"    Answer length: {len(result['answer'])} chars")
            print(f"    Sources: {len(result.get('sources', []))}")

    avg_time = sum(times) / len(times)

    print(f"\n📊 Results:")
    print(f"  Average end-to-end time: {avg_time:.2f}s")
    print(f"  Min: {min(times):.2f}s")
    print(f"  Max: {max(times):.2f}s")

    return avg_time


def measure_storage():
    """Measure storage requirements."""
    print("\n" + "="*80)
    print("BENCHMARK: Storage Requirements")
    print("="*80)

    def get_dir_size(path):
        """Get total size of directory in bytes."""
        total = 0
        try:
            for entry in Path(path).rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            print(f"Warning: Could not measure {path}: {e}")
        return total

    def format_size(bytes_size):
        """Format bytes to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

    data_dir = Path("data")

    components = {
        "Raw PDFs": data_dir / "raw_pdfs",
        "Processed Text": data_dir / "processed_text",
        "Chunks": data_dir / "chunks",
        "Vector Store": data_dir / "vector_store",
    }

    print("\n📊 Storage Breakdown:")
    total_size = 0

    for name, path in components.items():
        if path.exists():
            size = get_dir_size(path)
            total_size += size
            print(f"  {name:20} {format_size(size):>12}")
        else:
            print(f"  {name:20} {'NOT FOUND':>12}")

    print(f"  {'─'*20} {'─'*12}")
    print(f"  {'TOTAL':20} {format_size(total_size):>12}")

    # Check ChromaDB index specifically
    chromadb_dir = data_dir / "vector_store" / "chromadb"
    if chromadb_dir.exists():
        chromadb_size = get_dir_size(chromadb_dir)
        print(f"\n  ChromaDB Index: {format_size(chromadb_size)}")

    return total_size


def measure_memory():
    """Measure memory usage."""
    import psutil

    print("\n" + "="*80)
    print("BENCHMARK: Memory Usage")
    print("="*80)

    process = psutil.Process()

    # Measure current memory
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024

    print(f"\n📊 Current Process Memory:")
    print(f"  RSS (Resident Set Size): {mem_mb:.2f} MB")

    # Load the RAG system to measure peak
    print(f"\n  Loading RAG system...")
    from src.rag_pipeline.pipeline import RAGPipeline
    pipeline = RAGPipeline()

    mem_info_after = process.memory_info()
    mem_after_mb = mem_info_after.rss / 1024 / 1024

    print(f"  After loading RAG: {mem_after_mb:.2f} MB")
    print(f"  Increase: {mem_after_mb - mem_mb:.2f} MB")

    return mem_after_mb


def estimate_download_time():
    """Estimate PDF download time."""
    print("\n" + "="*80)
    print("ESTIMATE: PDF Download Time")
    print("="*80)

    # Check if PDFs exist
    pdf_dir = Path("data/raw_pdfs")
    if pdf_dir.exists():
        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        print(f"\nFound {pdf_count} existing PDFs in {pdf_dir}")
    else:
        pdf_count = 93  # Expected count

    print(f"\nEstimated download time for {pdf_count} PDFs:")
    print(f"  Average PDF size: ~500KB")
    print(f"  Download speed: ~1MB/s (typical)")
    print(f"  Network overhead: 0.5s per PDF (rate limiting)")

    # Calculation
    download_time = (pdf_count * 0.5)  # 0.5s per PDF (network + rate limit)
    data_transfer = (pdf_count * 0.5)  # 500KB at 1MB/s = 0.5s per PDF
    total_time = download_time + data_transfer

    print(f"\n📊 Breakdown:")
    print(f"  Network requests: {download_time:.1f}s ({pdf_count} × 0.5s)")
    print(f"  Data transfer: {data_transfer:.1f}s ({pdf_count} × 0.5MB @ 1MB/s)")
    print(f"  Total estimated: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    print(f"\n💡 Note: Actual time varies based on:")
    print(f"  - Internet speed")
    print(f"  - IRDAI server response time")
    print(f"  - Network congestion")
    print(f"  - Geographic location")

    return total_time


def generate_report():
    """Generate comprehensive benchmark report."""
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*80)

    report = {}

    # 0. Download time (estimate)
    try:
        report['download_time_s'] = estimate_download_time()
    except Exception as e:
        print(f"\n❌ Download time estimation failed: {e}")
        report['download_time_s'] = None

    # 1. Storage
    try:
        report['storage_bytes'] = measure_storage()
        report['storage_mb'] = report['storage_bytes'] / 1024 / 1024
    except Exception as e:
        print(f"\n❌ Storage benchmark failed: {e}")
        report['storage_mb'] = None

    # 2. Memory
    try:
        report['memory_mb'] = measure_memory()
    except Exception as e:
        print(f"\n❌ Memory benchmark failed: {e}")
        report['memory_mb'] = None

    # 3. Embedding time
    try:
        report['embedding_time_ms'] = benchmark_embeddings()
    except Exception as e:
        print(f"\n❌ Embedding benchmark failed: {e}")
        report['embedding_time_ms'] = None

    # 4. ChromaDB search time
    try:
        report['chromadb_time_ms'] = benchmark_chromadb_search()
    except Exception as e:
        print(f"\n❌ ChromaDB benchmark failed: {e}")
        report['chromadb_time_ms'] = None

    # 5. End-to-end query time
    try:
        report['query_time_s'] = benchmark_full_query()
    except Exception as e:
        print(f"\n❌ End-to-end benchmark failed: {e}")
        report['query_time_s'] = None

    # Print summary
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)

    # Build time section
    if report.get('download_time_s'):
        print(f"\n⏱️  Build Time (One-Time Setup):")
        print(f"  PDF Download: {report['download_time_s']/60:.1f} minutes")
        print(f"  PDF Processing: ~10-15 minutes (OCR + cleaning + chunking)")
        if report.get('embedding_time_ms'):
            embed_build_time = (501 * report['embedding_time_ms']) / 1000 / 60
            print(f"  Embedding Generation: {embed_build_time:.1f} minutes (501 chunks)")
        print(f"  Index Building: ~1-2 seconds")
        total_build = report['download_time_s']/60 + 12.5  # 12.5 = average of 10-15
        if report.get('embedding_time_ms'):
            total_build += embed_build_time
        print(f"  Total Build Time: ~{total_build:.0f} minutes")

    if report.get('storage_mb'):
        print(f"\n💾 Storage:")
        print(f"  Total: {report['storage_mb']:.2f} MB")

    if report.get('memory_mb'):
        print(f"\n🧠 Memory:")
        print(f"  Peak usage: {report['memory_mb']:.2f} MB")

    if report.get('embedding_time_ms'):
        print(f"\n⚡ Query Embedding Time:")
        print(f"  Average: {report['embedding_time_ms']:.2f} ms")

    if report.get('chromadb_time_ms'):
        print(f"\n🔍 ChromaDB Search Time:")
        print(f"  Average: {report['chromadb_time_ms']:.4f} ms")

    if report.get('query_time_s'):
        print(f"\n🎯 End-to-End Query Time:")
        print(f"  Average: {report['query_time_s']:.2f} seconds")

        # Break down the query time
        if report.get('embedding_time_ms') and report.get('chromadb_time_ms'):
            embedding_s = report['embedding_time_ms'] / 1000
            chromadb_s = report['chromadb_time_ms'] / 1000
            llm_s = report['query_time_s'] - embedding_s - chromadb_s

            print(f"\n  Breakdown:")
            print(f"    Embedding: {embedding_s:.3f}s ({embedding_s/report['query_time_s']*100:.1f}%)")
            print(f"    ChromaDB:  {chromadb_s:.3f}s ({chromadb_s/report['query_time_s']*100:.1f}%)")
            print(f"    LLM:       {llm_s:.3f}s ({llm_s/report['query_time_s']*100:.1f}%)")

    # Cost estimation
    print(f"\n💰 Cost Estimation:")
    provider = os.getenv('EMBEDDING_PROVIDER', 'sentence-transformers')
    if provider == 'openai':
        print(f"  Embedding provider: OpenAI")
        print(f"  Estimated cost per 1000 queries: ~$0.10")
        print(f"  Annual cost (1000 queries): ~$0.12")
    else:
        print(f"  Embedding provider: {provider}")
        print(f"  Cost: $0 (local)")

    llm_provider = os.getenv('LLM_PROVIDER', 'groq')
    print(f"  LLM provider: {llm_provider}")
    if llm_provider == 'groq':
        print(f"  LLM cost: $0 (free tier)")

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)

    return report


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory measurement...")
        os.system("pip install psutil")
        import psutil

    report = generate_report()
