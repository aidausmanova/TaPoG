import re
import string
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Callable

from .basic_utils import get_logger

logger = get_logger(__name__)


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def calculate_recall_k(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates Recall@k for each example and pools results for all queries.

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k_list (List[int]): List of k values to calculate Recall@k for.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A pooled dictionary with the averaged Recall@k across all examples.
            - A list of dictionaries with Recall@k for each example.
    """
    k_list = sorted(set(k_list))
    
    example_eval_results = []
    pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        if len(example_retrieved_docs) < k_list[-1]:
            logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})")
        
        example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}

        # Compute Recall@k for each k
        for k in k_list:
            # Get top-k retrieved documents
            top_k_docs = example_retrieved_docs[:k]
            # Calculate intersection with gold documents
            relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
            # Compute recall
            if example_gold_docs:  # Avoid division by zero
                example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
            else:
                example_eval_result[f"Recall@{k}"] = 0.0
        
        # Append example results
        example_eval_results.append(example_eval_result)
        
        # Accumulate pooled results
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

    # Average pooled results over all examples
    num_examples = len(gold_docs)
    for k in k_list:
        pooled_eval_results[f"Recall@{k}"] /= num_examples

    # round off to 4 decimal places for pooled results
    pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
    return pooled_eval_results, example_eval_results

def calculate_precision_k(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates Precision@k for each example and pools results for all queries.

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k_list (List[int]): List of k values to calculate Precision@k for.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A pooled dictionary with the averaged Precision@k across all examples.
            - A list of dictionaries with Precision@k for each example.
    """
    k_list = sorted(set(k_list))
    
    example_eval_results = []
    pooled_eval_results = {f"Precision@{k}": 0.0 for k in k_list}

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        if len(example_retrieved_docs) < k_list[-1]:
            logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for precision score ({k_list[-1]})")
        
        example_eval_result = {f"Precision@{k}": 0.0 for k in k_list}

        # Compute Precision@k for each k
        for k in k_list:
            # Get top-k retrieved documents
            top_k_docs = example_retrieved_docs[:k]
            # Calculate intersection with gold documents
            relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
            # Compute precision
            example_eval_result[f"Precision@{k}"] = len(relevant_retrieved) / k
        
        # Append example results
        example_eval_results.append(example_eval_result)
        
        # Accumulate pooled results
        for k in k_list:
            pooled_eval_results[f"Precision@{k}"] += example_eval_result[f"Precision@{k}"]

    # Average pooled results over all examples
    num_examples = len(gold_docs)
    for k in k_list:
        pooled_eval_results[f"Precision@{k}"] /= num_examples

    # round off to 4 decimal places for pooled results
    pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
    return pooled_eval_results, example_eval_results

def calculate_f1_k(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates F1@k for each example and pools results for all queries.

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k_list (List[int]): List of k values to calculate F1@k for.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A pooled dictionary with the averaged F1@k across all examples.
            - A list of dictionaries with F1@k for each example.
    """
    # Get precision and recall results
    precision_pooled, precision_examples = calculate_precision_k(gold_docs, retrieved_docs, k_list)
    recall_pooled, recall_examples = calculate_recall_k(gold_docs, retrieved_docs, k_list)
    
    k_list = sorted(set(k_list))
    
    example_eval_results = []
    pooled_eval_results = {f"F1@{k}": 0.0 for k in k_list}

    # Calculate F1 for each example
    for prec_result, rec_result in zip(precision_examples, recall_examples):
        example_eval_result = {}
        for k in k_list:
            precision = prec_result[f"Precision@{k}"]
            recall = rec_result[f"Recall@{k}"]
            
            # Calculate F1 score (harmonic mean)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            example_eval_result[f"F1@{k}"] = f1
        
        example_eval_results.append(example_eval_result)
        
        # Accumulate pooled results
        for k in k_list:
            pooled_eval_results[f"F1@{k}"] += example_eval_result[f"F1@{k}"]

    # Average pooled results over all examples
    num_examples = len(gold_docs)
    for k in k_list:
        pooled_eval_results[f"F1@{k}"] /= num_examples

    # round off to 4 decimal places for pooled results
    pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
    return pooled_eval_results, example_eval_results

def calculate_ndcg_k(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates NDCG@k for each example and pools results for all queries.
    
    NDCG formula:
    - DCG@k = sum(rel_i / log2(i+1)) for i in 1..k
    - IDCG@k = DCG of perfect ranking (all relevant docs at top)
    - NDCG@k = DCG@k / IDCG@k

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k_list (List[int]): List of k values to calculate NDCG@k for.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A pooled dictionary with the averaged NDCG@k across all examples.
            - A list of dictionaries with NDCG@k for each example.
    """
    k_list = sorted(set(k_list))
    
    example_eval_results = []
    pooled_eval_results = {f"NDCG@{k}": 0.0 for k in k_list}

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        if len(example_retrieved_docs) < k_list[-1]:
            logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for NDCG score ({k_list[-1]})")
        
        example_eval_result = {f"NDCG@{k}": 0.0 for k in k_list}
        gold_set = set(example_gold_docs)

        # Compute NDCG@k for each k
        for k in k_list:
            # Get top-k retrieved documents
            top_k_docs = example_retrieved_docs[:k]
            
            # Calculate DCG@k
            dcg = 0.0
            for i, doc in enumerate(top_k_docs):
                # Relevance is binary: 1 if doc is in gold_docs, 0 otherwise
                relevance = 1.0 if doc in gold_set else 0.0
                # Position is i+1 (1-indexed), discount factor is log2(i+2)
                dcg += relevance / np.log2(i + 2)
            
            # Calculate IDCG@k (Ideal DCG - if all relevant docs were at the top)
            # IDCG assumes we retrieve min(k, num_relevant_docs) relevant documents
            num_relevant = min(k, len(example_gold_docs))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
            
            # Calculate NDCG@k
            if idcg > 0:
                example_eval_result[f"NDCG@{k}"] = dcg / idcg
            else:
                example_eval_result[f"NDCG@{k}"] = 0.0
        
        # Append example results
        example_eval_results.append(example_eval_result)
        
        # Accumulate pooled results
        for k in k_list:
            pooled_eval_results[f"NDCG@{k}"] += example_eval_result[f"NDCG@{k}"]

    # Average pooled results over all examples
    num_examples = len(gold_docs)
    for k in k_list:
        pooled_eval_results[f"NDCG@{k}"] /= num_examples

    # round off to 4 decimal places for pooled results
    pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
    return pooled_eval_results, example_eval_results

def calculate_mrr(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k: int = None) -> Tuple[float, List[float]]:
    """
    Calculates Mean Reciprocal Rank (MRR) for each example and pools results for all queries.
    
    MRR = average of (1 / rank of first relevant document)
    If no relevant document is found (or beyond k), reciprocal rank is 0.

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k (int, optional): Consider only top-k documents. If None, considers all retrieved documents.

    Returns:
        Tuple[float, List[float]]: 
            - Pooled MRR across all examples.
            - List of reciprocal ranks for each example.
    """
    reciprocal_ranks = []

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        gold_set = set(example_gold_docs)
        
        # Limit to top-k if specified
        docs_to_check = example_retrieved_docs[:k] if k is not None else example_retrieved_docs
        
        # Find the rank of the first relevant document
        reciprocal_rank = 0.0
        for rank, doc in enumerate(docs_to_check, start=1):
            if doc in gold_set:
                reciprocal_rank = 1.0 / rank
                break
        
        reciprocal_ranks.append(reciprocal_rank)
    
    # Calculate mean reciprocal rank
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mrr = round(mrr, 4)
    
    return mrr, reciprocal_ranks

def calculate_exact_match(gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates the Exact Match (EM) score.

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.
        aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A dictionary with the averaged EM score.
            - A list of dictionaries with EM scores for each example.
    """
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    example_eval_results = []
    total_em = 0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
        aggregated_em = aggregation_fn(em_scores)
        example_eval_results.append({"ExactMatch": aggregated_em})
        total_em += aggregated_em

    avg_em = total_em / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"ExactMatch": avg_em}

    return pooled_eval_results, example_eval_results
