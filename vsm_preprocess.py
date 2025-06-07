import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Tải dữ liệu cần thiết từ nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') 

# Khởi tạo stemmer và stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Token hóa
    tokens = word_tokenize(text.lower())
    # Loại bỏ stopwords và stemming
    processed_tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(processed_tokens)

# Tải dữ liệu Cranfield
dataset = ir_datasets.load("cranfield")

# Lấy dữ liệu và tiền xử lý
documents = [preprocess_text(doc.text) for doc in dataset.docs_iter()]
queries = {query.query_id: preprocess_text(query.text) for query in dataset.queries_iter()}
query_ids = list(queries.keys())
query_texts = list(queries.values())

# Vector hóa với TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vectors = vectorizer.transform(query_texts)

# Tính độ tương tự cosine
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

# Hàm đánh giá (giữ nguyên như trước)
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

results = evaluate(rankings, qrels)
print(results)