import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import ir_datasets

dataset = ir_datasets.load("cranfield")

# Lấy dữ liệu
documents = [doc.text for doc in dataset.docs_iter()]
queries = {query.query_id: query.text for query in dataset.queries_iter()}  # Sử dụng từ điển
query_ids = list(queries.keys())
query_texts = list(queries.values())

# Vector hóa
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vectors = vectorizer.transform(query_texts)

# Tính độ tương tự
similarity = cosine_similarity(query_vectors, doc_vectors)

# Xếp hạng tài liệu
rankings = {}
for i, query_id in enumerate(query_ids):
    doc_scores = list(zip(range(len(documents)), similarity[i]))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    rankings[query_id] = [doc_id for doc_id, _ in doc_scores]

# Lấy qrels
qrels = defaultdict(list)
for qrel in dataset.qrels_iter():
    qrels[qrel.query_id].append(qrel.doc_id)

# Hàm đánh giá
def evaluate(rankings, qrels, k=10):
    precisions = []
    recalls = []
    aps = []

    for query_id, ranked_docs in rankings.items():
        relevant_docs = qrels.get(query_id, [])
        if not relevant_docs:
            continue  # Bỏ qua truy vấn không có tài liệu liên quan

        # Precision và Recall tại k
        retrieved = ranked_docs[:k]
        relevant_retrieved = [doc for doc in retrieved if str(doc) in relevant_docs]
        precision = len(relevant_retrieved) / k if k > 0 else 0.0
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
        precisions.append(precision)
        recalls.append(recall)

        # Average Precision
        ap = 0.0
        num_relevant = 0
        for i, doc in enumerate(ranked_docs):
            if str(doc) in relevant_docs:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        ap /= len(relevant_docs) if relevant_docs else 1.0
        aps.append(ap)

    # 11-point interpolated precision
    interpolated_precision = []
    for r in np.linspace(0, 1, 11):
        max_p = max([p for p, rec in zip(precisions, recalls) if rec >= r], default=0.0)
        interpolated_precision.append(max_p)

    return {
        "Precision@k": np.mean(precisions) if precisions else 0.0,
        "Recall@k": np.mean(recalls) if recalls else 0.0,
        "MAP": np.mean(aps) if aps else 0.0,
        "11-point Interpolated Precision": interpolated_precision,
    }

results = evaluate(rankings, qrels)
print(results)