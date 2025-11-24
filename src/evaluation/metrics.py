"""
RAG Evaluation Metrics

Implements the following metrics:
1. Retrieval Metrics:
   - Mean Reciprocal Rank (MRR)
   - Hit Rate @ K

2. RAG-Specific Metrics:
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Relevancy
"""

import re
from typing import List, Dict, Any
import numpy as np


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR measures how high the first relevant document appears in the ranking.
        MRR = 1 / rank of first relevant document

        Args:
            retrieved_docs: List of retrieved document names in order
            relevant_docs: List of ground truth relevant document names

        Returns:
            MRR score (0 to 1, higher is better)
        """
        for rank, doc in enumerate(retrieved_docs, start=1):
            if any(relevant_doc in doc or doc in relevant_doc for relevant_doc in relevant_docs):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def hit_rate_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Calculate Hit Rate @ K (also called Recall @ K).

        Hit Rate @ K = 1 if at least one relevant doc is in top K results, else 0

        Args:
            retrieved_docs: List of retrieved document names
            relevant_docs: List of ground truth relevant document names
            k: Number of top results to consider

        Returns:
            Hit rate (0 or 1)
        """
        top_k_docs = retrieved_docs[:k]
        for doc in top_k_docs:
            if any(relevant_doc in doc or doc in relevant_doc for relevant_doc in relevant_docs):
                return 1.0
        return 0.0

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Calculate Precision @ K.

        Precision @ K = (# relevant docs in top K) / K

        Args:
            retrieved_docs: List of retrieved document names
            relevant_docs: List of ground truth relevant document names
            k: Number of top results to consider

        Returns:
            Precision score (0 to 1)
        """
        top_k_docs = retrieved_docs[:k]
        relevant_count = sum(
            1 for doc in top_k_docs
            if any(relevant_doc in doc or doc in relevant_doc for relevant_doc in relevant_docs)
        )
        return relevant_count / k if k > 0 else 0.0


class RAGMetrics:
    """Metrics for evaluating RAG system quality."""

    @staticmethod
    def context_precision(context: str, answer: str, relevant_keywords: List[str]) -> float:
        """
        Calculate Context Precision.

        Context Precision measures how much of the retrieved context is actually
        relevant to answering the question. We approximate this by checking if
        the context contains keywords that appear in the ground truth answer.

        Args:
            context: Retrieved context
            answer: Ground truth answer
            relevant_keywords: Keywords that should be in relevant context

        Returns:
            Context precision score (0 to 1)
        """
        context_lower = context.lower()

        # Count how many relevant keywords appear in the context
        keyword_matches = sum(1 for kw in relevant_keywords if kw.lower() in context_lower)

        if len(relevant_keywords) == 0:
            return 1.0

        return keyword_matches / len(relevant_keywords)

    @staticmethod
    def context_recall(context: str, ground_truth: str, relevant_keywords: List[str]) -> float:
        """
        Calculate Context Recall.

        Context Recall measures if all necessary information from ground truth
        is present in the retrieved context.

        Args:
            context: Retrieved context
            ground_truth: Ground truth answer
            relevant_keywords: Keywords from ground truth that should be in context

        Returns:
            Context recall score (0 to 1)
        """
        context_lower = context.lower()

        # Check if context contains the critical information
        keyword_in_context = sum(1 for kw in relevant_keywords if kw.lower() in context_lower)

        if len(relevant_keywords) == 0:
            return 1.0

        return keyword_in_context / len(relevant_keywords)

    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """
        Calculate Faithfulness (Answer Groundedness).

        Faithfulness measures if the answer is grounded in the provided context
        (i.e., doesn't hallucinate information not in context).

        We approximate this by:
        1. Extracting key claims/facts from the answer
        2. Checking if those claims appear in the context

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Faithfulness score (0 to 1, higher means less hallucination)
        """
        if not answer or not context:
            return 0.0

        # Extract sentences from answer
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) == 0:
            return 1.0

        context_lower = context.lower()

        # Check how many sentences have supporting evidence in context
        supported_sentences = 0
        for sentence in sentences:
            # Extract key terms from sentence (words longer than 3 chars)
            key_terms = [
                word.lower() for word in re.findall(r'\b\w+\b', sentence)
                if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 'must', 'shall']
            ]

            # If at least 50% of key terms appear in context, consider it supported
            if key_terms:
                term_matches = sum(1 for term in key_terms if term in context_lower)
                if term_matches / len(key_terms) >= 0.5:
                    supported_sentences += 1

        return supported_sentences / len(sentences)

    @staticmethod
    def answer_relevancy(answer: str, question: str, ground_truth: str = None) -> float:
        """
        Calculate Answer Relevancy.

        Answer Relevancy measures how relevant the answer is to the question.
        We approximate this by:
        1. Checking if answer addresses key terms from question
        2. If ground truth is available, checking similarity to ground truth

        Args:
            answer: Generated answer
            question: User's question
            ground_truth: Optional ground truth answer for comparison

        Returns:
            Answer relevancy score (0 to 1)
        """
        if not answer:
            return 0.0

        answer_lower = answer.lower()
        question_lower = question.lower()

        # Extract key terms from question (excluding common question words)
        question_stopwords = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 'for', 'in', 'on', 'at', 'to', 'and', 'or'}
        question_terms = [
            word.lower() for word in re.findall(r'\b\w+\b', question)
            if len(word) > 3 and word.lower() not in question_stopwords
        ]

        # Check how many question terms appear in the answer
        relevancy_score = 0.0
        if question_terms:
            term_matches = sum(1 for term in question_terms if term in answer_lower)
            relevancy_score = term_matches / len(question_terms)

        # If ground truth is available, check similarity
        if ground_truth:
            ground_truth_lower = ground_truth.lower()

            # Extract key facts from ground truth
            gt_terms = [
                word.lower() for word in re.findall(r'\b\w+\b', ground_truth)
                if len(word) > 3 and word.lower() not in question_stopwords
            ]

            if gt_terms:
                gt_matches = sum(1 for term in gt_terms if term in answer_lower)
                gt_similarity = gt_matches / len(gt_terms)

                # Average question relevancy and ground truth similarity
                relevancy_score = (relevancy_score + gt_similarity) / 2

        return relevancy_score

    @staticmethod
    def semantic_similarity(text1: str, text2: str, embedder=None) -> float:
        """
        Calculate semantic similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text
            embedder: Optional embedding model (if None, uses keyword overlap)

        Returns:
            Similarity score (0 to 1)
        """
        if embedder is not None:
            # Use embedding-based similarity
            try:
                emb1 = embedder.embed_text(text1)
                emb2 = embedder.embed_text(text2)

                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity)
            except:
                pass

        # Fallback to keyword overlap
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


def evaluate_rag_response(
    question: str,
    answer: str,
    context: str,
    retrieved_docs: List[str],
    ground_truth_answer: str,
    relevant_docs: List[str],
    context_keywords: List[str],
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluate a single RAG response with all metrics.

    Args:
        question: User's question
        answer: Generated answer
        context: Retrieved context
        retrieved_docs: List of retrieved document names
        ground_truth_answer: Expected answer
        relevant_docs: List of relevant document names
        context_keywords: Keywords that should be in context
        k: Number of top results for Hit Rate @ K

    Returns:
        Dictionary of metric scores
    """
    retrieval = RetrievalMetrics()
    rag = RAGMetrics()

    return {
        # Retrieval metrics
        'mrr': retrieval.mean_reciprocal_rank(retrieved_docs, relevant_docs),
        'hit_rate@5': retrieval.hit_rate_at_k(retrieved_docs, relevant_docs, k=5),
        'hit_rate@10': retrieval.hit_rate_at_k(retrieved_docs, relevant_docs, k=10),
        'precision@5': retrieval.precision_at_k(retrieved_docs, relevant_docs, k=5),

        # RAG metrics
        'context_precision': rag.context_precision(context, answer, context_keywords),
        'context_recall': rag.context_recall(context, ground_truth_answer, context_keywords),
        'faithfulness': rag.faithfulness(answer, context),
        'answer_relevancy': rag.answer_relevancy(answer, question, ground_truth_answer),

        # Similarity metrics
        'answer_similarity': rag.semantic_similarity(answer, ground_truth_answer)
    }
