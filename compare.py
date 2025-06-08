import bm25
import vsm.vsm as vsm
import numpy as np

def print_results(model_name, results):
    print(f"\n=== Kết quả đánh giá cho mô hình {model_name} ===")
    print(f"Precision@10: {results['Precision@k']:.4f}")
    print(f"Recall@10: {results['Recall@k']:.4f}")
    print(f"MAP: {results['MAP']:.4f}")
    print("11-point Interpolated Precision:")
    for i, p in enumerate(results['11-point Interpolated Precision']):
        print(f"  Recall={i/10:.1f}: {p:.4f}")

# Đánh giá BM25
bm25_results = bm25.evaluate(bm25.rankings, bm25.qrels)
print_results("BM25", bm25_results)

# Đánh giá VSM
vsm_results = vsm.evaluate(vsm.rankings, vsm.qrels)
print_results("VSM", vsm_results)

# So sánh MAP
print("\n=== So sánh MAP giữa hai mô hình ===")
print(f"BM25 MAP: {bm25_results['MAP']:.4f}")
print(f"VSM MAP: {vsm_results['MAP']:.4f}")
if bm25_results['MAP'] > vsm_results['MAP']:
    print("BM25 hoạt động tốt hơn VSM trên ngữ liệu Cranfield.")
else:
    print("VSM hoạt động tốt hơn BM25 trên ngữ liệu Cranfield.")