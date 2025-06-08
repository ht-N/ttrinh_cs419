import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
import ir_datasets
from nltk.tokenize import RegexpTokenizer

# Token hóa giữ lại ký hiệu
tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

def preprocess(text):
    return tokenizer.tokenize(text.lower())

# Tải dữ liệu
dataset = ir_datasets.load("cranfield")
documents = [doc.text for doc in dataset.docs_iter()]
queries = {query.query_id: query.text for query in dataset.queries_iter()}
query_ids = list(queries.keys())

# Tiền xử lý
tokenized_docs = [preprocess(doc) for doc in documents]
tokenized_queries = [preprocess(queries[qid]) for qid in query_ids]

# BM25 với tham số tối ưu
bm25 = BM25Okapi(tokenized_docs, k1=0.9, b=0.3)  # Điều chỉnh tại đây

# Xếp hạng
rankings = {}
for i, query_id in enumerate(query_ids):
    scores = bm25.get_scores(tokenized_queries[i])
    doc_scores = list(zip(range(len(documents)), scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    rankings[query_id] = [str(doc_id) for doc_id, _ in doc_scores]

# Qrels (đảm bảo doc_id là chuỗi)
qrels = defaultdict(list)
for qrel in dataset.qrels_iter():
    qrels[qrel.query_id].append(str(qrel.doc_id))
# Hàm đánh giá (giữ nguyên)
def evaluate(rankings, qrels, k=10):
    precisions = []
    recalls = []
    aps = []

    for query_id, ranked_docs in rankings.items():
        relevant_docs = qrels.get(query_id, [])
        if not relevant_docs:
            continue

        retrieved = ranked_docs[:k]
        relevant_retrieved = [doc for doc in retrieved if str(doc) in relevant_docs]
        precision = len(relevant_retrieved) / k if k > 0 else 0.0
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
        precisions.append(precision)
        recalls.append(recall)

        ap = 0.0
        num_relevant = 0
        for i, doc in enumerate(ranked_docs):
            if str(doc) in relevant_docs:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        ap /= len(relevant_docs) if relevant_docs else 1.0
        aps.append(ap)

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

sample_qid = list(qrels.keys())[0]
print(f"Debug - Query ID: {sample_qid}")
print(f"Query text: {queries[sample_qid]}")
print(f"Relevant docs (qrels): {qrels[sample_qid]}")
print(f"Top 5 ranked docs: {rankings[sample_qid][:5]}")

results = evaluate(rankings, qrels)
print("Final Results:", results)