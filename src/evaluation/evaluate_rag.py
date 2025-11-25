"""
Comprehensive RAG System Evaluation

This script evaluates the RAG system using:
1. Retrieval Metrics: MRR, Hit Rate @ K, Precision @ K
2. RAG Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy

Usage:
    python evaluate_rag.py
    python evaluate_rag.py --output results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

# Add src to path - go up to project root, then add src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from rag_pipeline.pipeline import RAGPipeline
from evaluation.metrics import evaluate_rag_response


def load_evaluation_dataset(path: str = None) -> List[Dict]:
    """Load the evaluation dataset."""
    if path is None:
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "data" / "evaluation_dataset.json"

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_single_query(
    pipeline: RAGPipeline,
    eval_item: Dict,
    top_k: int = 5
) -> Dict:
    """
    Evaluate a single query.

    Args:
        pipeline: RAG pipeline
        eval_item: Evaluation item with question and ground truth
        top_k: Number of documents to retrieve

    Returns:
        Evaluation results with metrics
    """
    question = eval_item['question']
    ground_truth_answer = eval_item['ground_truth_answer']
    relevant_docs = eval_item['relevant_docs']
    context_keywords = eval_item.get('context_keywords', [])

    # Query the RAG system
    result = pipeline.query(question, top_k=top_k)

    # Extract retrieved document names
    retrieved_docs = result.get('sources', [])

    # Get generated answer and context
    answer = result.get('answer', '')
    context = result.get('context', '')

    # Evaluate metrics
    metrics = evaluate_rag_response(
        question=question,
        answer=answer,
        context=context,
        retrieved_docs=retrieved_docs,
        ground_truth_answer=ground_truth_answer,
        relevant_docs=relevant_docs,
        context_keywords=context_keywords,
        k=top_k
    )

    return {
        'question': question,
        'answer': answer,
        'ground_truth': ground_truth_answer,
        'retrieved_docs': retrieved_docs,
        'relevant_docs': relevant_docs,
        'metrics': metrics,
        'llm_provider': result.get('llm_provider'),
        'llm_model': result.get('llm_model')
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """
    Aggregate metrics across all queries.

    Args:
        results: List of evaluation results

    Returns:
        Aggregated metrics with mean, std, min, max
    """
    # Collect all metrics
    all_metrics = {}
    for result in results:
        for metric_name, value in result['metrics'].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute statistics
    aggregated = {}
    for metric_name, values in all_metrics.items():
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    return aggregated


def print_results_table(aggregated: Dict):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Group metrics
    retrieval_metrics = ['mrr', 'hit_rate@5', 'hit_rate@10', 'precision@5']
    rag_metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy', 'answer_similarity']

    print("\n[RETRIEVAL METRICS]")
    print("-" * 80)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)

    for metric in retrieval_metrics:
        if metric in aggregated:
            stats = aggregated[metric]
            print(f"{metric:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}")

    print("\n[RAG QUALITY METRICS]")
    print("-" * 80)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)

    for metric in rag_metrics:
        if metric in aggregated:
            stats = aggregated[metric]
            print(f"{metric:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}")

    print("\n" + "="*80)


def generate_report(results: List[Dict], aggregated: Dict, output_path: str = None):
    """
    Generate comprehensive evaluation report.

    Args:
        results: Individual evaluation results
        aggregated: Aggregated metrics
        output_path: Optional path to save JSON report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_queries': len(results),
        'aggregated_metrics': aggregated,
        'individual_results': results
    }

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Detailed report saved to: {output_path}")

    return report


def print_top_bottom_performers(results: List[Dict]):
    """Print top and bottom performing queries."""
    # Sort by answer relevancy
    sorted_results = sorted(results, key=lambda x: x['metrics']['answer_relevancy'], reverse=True)

    print("\n" + "="*80)
    print("[TOP 3] PERFORMING QUERIES (by Answer Relevancy)")
    print("="*80)

    for i, result in enumerate(sorted_results[:3], 1):
        print(f"\n{i}. {result['question']}")
        print(f"   Answer Relevancy: {result['metrics']['answer_relevancy']:.4f}")
        print(f"   Faithfulness: {result['metrics']['faithfulness']:.4f}")
        print(f"   MRR: {result['metrics']['mrr']:.4f}")

    print("\n" + "="*80)
    print("[BOTTOM 3] QUERIES NEEDING IMPROVEMENT")
    print("="*80)

    for i, result in enumerate(sorted_results[-3:], 1):
        print(f"\n{i}. {result['question']}")
        print(f"   Answer Relevancy: {result['metrics']['answer_relevancy']:.4f}")
        print(f"   Faithfulness: {result['metrics']['faithfulness']:.4f}")
        print(f"   MRR: {result['metrics']['mrr']:.4f}")
        print(f"   Generated Answer: {result['answer'][:150]}...")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate RAG system')
    parser.add_argument('--dataset', default=None, help='Path to evaluation dataset')
    parser.add_argument('--output', default=None, help='Path to save results (JSON)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of documents to retrieve')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')

    args = parser.parse_args()

    print("="*80)
    print("RAG SYSTEM EVALUATION")
    print("="*80)

    # Load evaluation dataset
    if args.dataset:
        print(f"\nLoading evaluation dataset from: {args.dataset}")
    else:
        project_root = Path(__file__).parent.parent.parent
        args.dataset = str(project_root / "data" / "evaluation_dataset.json")
        print(f"\nLoading evaluation dataset from: {args.dataset}")

    eval_dataset = load_evaluation_dataset(args.dataset)
    print(f"[OK] Loaded {len(eval_dataset)} evaluation queries")

    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline()
    print("[OK] Pipeline initialized")

    # Evaluate each query
    print(f"\nEvaluating {len(eval_dataset)} queries...")
    print("-" * 80)

    results = []
    total_time = 0

    for i, eval_item in enumerate(eval_dataset, 1):
        question = eval_item['question']
        print(f"\n[{i}/{len(eval_dataset)}] {question[:60]}...")

        start_time = time.time()
        result = evaluate_single_query(pipeline, eval_item, top_k=args.top_k)
        elapsed = time.time() - start_time
        total_time += elapsed

        results.append(result)

        # Print key metrics
        metrics = result['metrics']
        print(f"  MRR: {metrics['mrr']:.3f} | Hit@5: {metrics['hit_rate@5']:.3f} | "
              f"Faithfulness: {metrics['faithfulness']:.3f} | Answer Relevancy: {metrics['answer_relevancy']:.3f}")
        print(f"  Time: {elapsed:.2f}s")

    print("\n" + "="*80)
    print(f"[OK] Evaluation complete! Total time: {total_time:.2f}s (avg: {total_time/len(eval_dataset):.2f}s/query)")

    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = aggregate_metrics(results)

    # Print results
    print_results_table(aggregated)
    print_top_bottom_performers(results)

    # Generate report
    output_path = args.output or f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = generate_report(results, aggregated, output_path)

    # Print summary insights
    print("\n" + "="*80)
    print("[KEY INSIGHTS]")
    print("="*80)

    insights = []

    # Check retrieval quality
    mrr_mean = aggregated['mrr']['mean']
    if mrr_mean > 0.8:
        insights.append(f"[EXCELLENT] Retrieval quality (MRR: {mrr_mean:.3f})")
    elif mrr_mean > 0.5:
        insights.append(f"[GOOD] Retrieval quality, can be improved (MRR: {mrr_mean:.3f})")
    else:
        insights.append(f"[POOR] Retrieval quality (MRR: {mrr_mean:.3f}) - consider improving embeddings or chunking")

    # Check faithfulness
    faith_mean = aggregated['faithfulness']['mean']
    if faith_mean > 0.8:
        insights.append(f"[EXCELLENT] High faithfulness - minimal hallucination (Faithfulness: {faith_mean:.3f})")
    elif faith_mean > 0.6:
        insights.append(f"[MODERATE] Faithfulness (Faithfulness: {faith_mean:.3f}) - some hallucination detected")
    else:
        insights.append(f"[LOW] Faithfulness (Faithfulness: {faith_mean:.3f}) - significant hallucination")

    # Check answer relevancy
    rel_mean = aggregated['answer_relevancy']['mean']
    if rel_mean > 0.7:
        insights.append(f"[EXCELLENT] Answer relevancy (Answer Relevancy: {rel_mean:.3f})")
    else:
        insights.append(f"[MODERATE] Answer relevancy can be improved (Answer Relevancy: {rel_mean:.3f})")

    # Check context recall
    recall_mean = aggregated['context_recall']['mean']
    if recall_mean < 0.7:
        insights.append(f"[WARNING] Low context recall ({recall_mean:.3f}) - important information may be missed in retrieval")

    for insight in insights:
        print(f"\n{insight}")

    print("\n" + "="*80)
    print("[COMPLETE] EVALUATION FINISHED")
    print("="*80)


if __name__ == "__main__":
    main()
