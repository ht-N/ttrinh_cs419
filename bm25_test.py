import numpy as np
from rank_bm25 import BM25Okapi
import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load dataset
dataset = ir_datasets.load("cranfield")

def preprocess_text_bm25(text, method='stemming'):
    """
    Tiền xử lý văn bản cho BM25
    Returns: list of tokens
    """
    text = text.lower()
    
    if method == 'basic':
        tokens = word_tokenize(text)
        return [token for token in tokens if token.isalpha()]
    
    elif method == 'stopwords':
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        return filtered_tokens
    
    elif method == 'stemming':
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        return stemmed_tokens
    
    elif method == 'lemmatization':
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        return lemmatized_tokens

def calculate_interpolated_precision_bm25(recall_levels, precisions, recalls):
    """
    Tính precision nội suy tại các mức recall chuẩn
    """
    interpolated_precisions = []
    
    for target_recall in recall_levels:
        valid_precisions = [p for p, r in zip(precisions, recalls) if r >= target_recall]
        
        if valid_precisions:
            interpolated_precisions.append(max(valid_precisions))
        else:
            interpolated_precisions.append(0.0)
    
    return interpolated_precisions

def calculate_11_point_map_bm25(ranked_docs, relevant_docs):
    """
    Tính MAP 11-point interpolated theo chuẩn TREC
    """
    if not relevant_docs:
        return 0.0, [0.0] * 11
    
    recall_levels = [i * 0.1 for i in range(11)]
    precisions = []
    recalls = []
    relevant_retrieved = 0
    
    for i, doc_id in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            relevant_retrieved += 1
            precision = relevant_retrieved / (i + 1)
            recall = relevant_retrieved / len(relevant_docs)
            precisions.append(precision)
            recalls.append(recall)
    
    if not precisions:
        return 0.0, [0.0] * 11
    
    interpolated_precisions = calculate_interpolated_precision_bm25(recall_levels, precisions, recalls)
    map_score = sum(interpolated_precisions) / len(interpolated_precisions)
    
    return map_score, interpolated_precisions

def evaluate_bm25_trec(doc_texts, query_texts, qrels, preprocessing_method='stemming', top_k=1000):
    """
    Đánh giá BM25 với TREC 11-point interpolated MAP
    """
    processed_docs = [preprocess_text_bm25(text, preprocessing_method) for text in doc_texts]
    processed_queries = [preprocess_text_bm25(text, preprocessing_method) for text in query_texts]
    
    bm25 = BM25Okapi(processed_docs)
    
    all_precisions_at_10 = []
    all_recalls_at_10 = []
    all_map_scores = []
    all_interpolated_precisions = []
    
    for i, query_tokens in enumerate(processed_queries):
        scores = bm25.get_scores(query_tokens)
        ranked_docs = np.argsort(scores)[::-1][:top_k]
        
        if isinstance(qrels, dict):
            relevant_docs = qrels.get(i, set())
        else:
            query_id = list(qrels.keys())[i] if i < len(qrels) else None
            relevant_docs = qrels.get(query_id, set()) if query_id else set()
        
        if len(relevant_docs) == 0:
            continue
        
        top_10_docs = set(ranked_docs[:10])
        retrieved_relevant_10 = top_10_docs.intersection(relevant_docs)
        
        precision_10 = len(retrieved_relevant_10) / 10 if len(top_10_docs) > 0 else 0
        recall_10 = len(retrieved_relevant_10) / len(relevant_docs) if len(relevant_docs) > 0 else 0
        
        all_precisions_at_10.append(precision_10)
        all_recalls_at_10.append(recall_10)
        
        map_score, interpolated_precs = calculate_11_point_map_bm25(ranked_docs, relevant_docs)
        all_map_scores.append(map_score)
        all_interpolated_precisions.append(interpolated_precs)
    
    avg_precision_10 = np.mean(all_precisions_at_10) if all_precisions_at_10 else 0
    avg_recall_10 = np.mean(all_recalls_at_10) if all_recalls_at_10 else 0
    avg_f1_10 = 2 * avg_precision_10 * avg_recall_10 / (avg_precision_10 + avg_recall_10) if (avg_precision_10 + avg_recall_10) > 0 else 0
    avg_map_11point = np.mean(all_map_scores) if all_map_scores else 0
    
    avg_interpolated_precisions = []
    if all_interpolated_precisions:
        for j in range(11):
            recall_level_precs = [precs[j] for precs in all_interpolated_precisions]
            avg_interpolated_precisions.append(np.mean(recall_level_precs))
    else:
        avg_interpolated_precisions = [0.0] * 11
    
    return {
        'precision_10': avg_precision_10,
        'recall_10': avg_recall_10,
        'f1_10': avg_f1_10,
        'map_11point': avg_map_11point,
        'interpolated_precisions': avg_interpolated_precisions,
        'num_queries': len(all_map_scores)
    }

def main():
    print("Đang preprocessing text...")
    
    # Load all data
    all_docs = list(dataset.docs_iter())
    all_queries = list(dataset.queries_iter())
    all_qrels_raw = list(dataset.qrels_iter())
    
    # Prepare texts
    all_doc_texts = [(doc.title + " " + doc.text).strip() for doc in all_docs]
    all_query_texts = [query.text for query in all_queries]
    all_doc_ids = [doc.doc_id for doc in all_docs]
    all_query_ids = [query.query_id for query in all_queries]
    
    # Create qrels mapping
    all_qrels = {}
    for qrel in all_qrels_raw:
        if qrel.query_id not in all_qrels:
            all_qrels[qrel.query_id] = set()
        all_qrels[qrel.query_id].add(qrel.doc_id)
    
    # Create subset (first 100 docs, 10 queries)
    subset_doc_texts = all_doc_texts[:100]
    subset_query_texts = all_query_texts[:10]
    subset_doc_ids = all_doc_ids[:100]
    subset_query_ids = all_query_ids[:10]
    
    # Create mappings
    subset_doc_id_to_index = {doc_id: i for i, doc_id in enumerate(subset_doc_ids)}
    subset_query_id_to_index = {query_id: i for i, query_id in enumerate(subset_query_ids)}
    
    all_doc_id_to_index = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
    all_query_id_to_index = {query_id: i for i, query_id in enumerate(all_query_ids)}
    
    # Convert qrels for subset
    subset_qrels_indexed = {}
    for query_id, doc_ids in all_qrels.items():
        if query_id in subset_query_id_to_index:
            query_index = subset_query_id_to_index[query_id]
            subset_qrels_indexed[query_index] = set()
            for doc_id in doc_ids:
                if doc_id in subset_doc_id_to_index:
                    doc_index = subset_doc_id_to_index[doc_id]
                    subset_qrels_indexed[query_index].add(doc_index)
    
    # Convert qrels for full dataset
    all_qrels_indexed = {}
    for query_id, doc_ids in all_qrels.items():
        if query_id in all_query_id_to_index:
            query_index = all_query_id_to_index[query_id]
            all_qrels_indexed[query_index] = set()
            for doc_id in doc_ids:
                if doc_id in all_doc_id_to_index:
                    doc_index = all_doc_id_to_index[doc_id]
                    all_qrels_indexed[query_index].add(doc_index)
    
    # Test on subset
    print("Test trên subset Cranfield dataset...")
    bm25_subset_result = evaluate_bm25_trec(
        subset_doc_texts, 
        subset_query_texts, 
        subset_qrels_indexed, 
        preprocessing_method='stemming'
    )
    
    # Test on full dataset
    print("Test trên full Cranfield dataset...")
    bm25_full_result = evaluate_bm25_trec(
        all_doc_texts, 
        all_query_texts, 
        all_qrels_indexed, 
        preprocessing_method='stemming'
    )
    
    # Print results
    print("\n" + "="*80)
    print("KẾT QUẢ BM25 (STEMMING) - TREC 11-POINT INTERPOLATED MAP")
    print("="*80)
    
    print(f"\nKẾT QUẢ SUBSET (100 docs, 10 queries):")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 35)
    print(f"{'Precision@10':<20} {bm25_subset_result['precision_10']:<10.4f}")
    print(f"{'Recall@10':<20} {bm25_subset_result['recall_10']:<10.4f}")
    print(f"{'F1@10':<20} {bm25_subset_result['f1_10']:<10.4f}")
    print(f"{'MAP 11-point':<20} {bm25_subset_result['map_11point']:<10.4f}")
    print(f"{'Queries processed':<20} {bm25_subset_result['num_queries']:<10}")
    
    print(f"\nKẾT QUẢ FULL DATASET ({len(all_doc_texts)} docs, {len(all_query_texts)} queries):")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 35)
    print(f"{'Precision@10':<20} {bm25_full_result['precision_10']:<10.4f}")
    print(f"{'Recall@10':<20} {bm25_full_result['recall_10']:<10.4f}")
    print(f"{'F1@10':<20} {bm25_full_result['f1_10']:<10.4f}")
    print(f"{'MAP 11-point':<20} {bm25_full_result['map_11point']:<10.4f}")
    print(f"{'Queries processed':<20} {bm25_full_result['num_queries']:<10}")
    
    print(f"\nSO SÁNH SUBSET vs FULL DATASET:")
    print(f"{'Metric':<20} {'Subset':<10} {'Full':<10} {'Δ (Full-Subset)':<15}")
    print("-" * 60)
    
    metrics_compare = ['precision_10', 'recall_10', 'f1_10', 'map_11point']
    metric_names = ['Precision@10', 'Recall@10', 'F1@10', 'MAP 11-point']
    
    for metric, name in zip(metrics_compare, metric_names):
        subset_val = bm25_subset_result[metric]
        full_val = bm25_full_result[metric]
        delta = full_val - subset_val
        print(f"{name:<20} {subset_val:<10.4f} {full_val:<10.4f} {delta:+.4f}")
    
    print("-" * 60)

if __name__ == "__main__":
    main() 