import subprocess
import sys
import time

def run_model_test(model_name, script_name):
    """Chạy test cho một mô hình và capture output"""
    print(f"\n{'='*80}")
    print(f"ĐANG CHẠY {model_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✅ {model_name} hoàn thành trong {execution_time:.2f} giây")
            return result.stdout, execution_time, True
        else:
            print(f"❌ Lỗi khi chạy {model_name}:")
            print(result.stderr)
            return None, execution_time, False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} vượt quá thời gian cho phép (5 phút)")
        return None, 300, False
    except Exception as e:
        print(f"❌ Lỗi không xác định khi chạy {model_name}: {e}")
        return None, 0, False

def extract_metrics_from_output(output):
    """Trích xuất các metrics từ output"""
    if not output:
        return None
    
    lines = output.split('\n')
    metrics = {}
    
    # Tìm phần kết quả full dataset
    in_full_section = False
    for line in lines:
        if "KẾT QUẢ FULL DATASET" in line:
            in_full_section = True
            continue
        elif "SO SÁNH SUBSET vs FULL DATASET" in line:
            break
        elif in_full_section and line.strip() and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ['Precision@10', 'Recall@10', 'F1@10', 'MAP']:
                try:
                    if parts[0] == 'MAP':
                        metrics['MAP_11point'] = float(parts[1])
                    else:
                        metrics[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    
    return metrics

def main():
    print("🚀 BẮT ĐẦU SO SÁNH VSM VÀ BM25")
    print(f"{'='*80}")
    print("Chương trình sẽ chạy lần lượt VSM và BM25, sau đó so sánh kết quả")
    print(f"{'='*80}")
    
    # Chạy VSM
    vsm_output, vsm_time, vsm_success = run_model_test("VSM", "vsm_test.py")
    
    # Chạy BM25
    bm25_output, bm25_time, bm25_success = run_model_test("BM25", "bm25_test.py")
    
    # So sánh kết quả
    print(f"\n{'='*80}")
    print("SO SÁNH TỔNG QUAN VSM vs BM25")
    print(f"{'='*80}")
    
    if vsm_success and bm25_success:
        vsm_metrics = extract_metrics_from_output(vsm_output)
        bm25_metrics = extract_metrics_from_output(bm25_output)
        
        if vsm_metrics and bm25_metrics:
            print(f"\n📊 BẢNG SO SÁNH HIỆU SUẤT (FULL DATASET):")
            print(f"{'Metric':<15} {'VSM':<12} {'BM25':<12} {'Δ (VSM-BM25)':<15} {'Winner':<10}")
            print("-" * 70)
            
            metric_names = ['Precision@10', 'Recall@10', 'F1@10', 'MAP_11point']
            display_names = ['Precision@10', 'Recall@10', 'F1@10', 'MAP 11-point']
            
            for metric, display in zip(metric_names, display_names):
                if metric in vsm_metrics and metric in bm25_metrics:
                    vsm_val = vsm_metrics[metric]
                    bm25_val = bm25_metrics[metric]
                    delta = vsm_val - bm25_val
                    winner = "VSM" if delta > 0 else "BM25" if delta < 0 else "TIE"
                    winner_symbol = "🏆" if winner != "TIE" else "🤝"
                    
                    print(f"{display:<15} {vsm_val:<12.4f} {bm25_val:<12.4f} {delta:+.4f}          {winner_symbol} {winner}")
            
            print("-" * 70)
            
            # Tổng kết
            vsm_wins = sum(1 for metric in metric_names 
                          if metric in vsm_metrics and metric in bm25_metrics 
                          and vsm_metrics[metric] > bm25_metrics[metric])
            bm25_wins = sum(1 for metric in metric_names 
                           if metric in vsm_metrics and metric in bm25_metrics 
                           and bm25_metrics[metric] > vsm_metrics[metric])
            
            print(f"\n🏆 TỔNG KẾT:")
            print(f"VSM thắng:  {vsm_wins} metrics")
            print(f"BM25 thắng: {bm25_wins} metrics")
            
            if vsm_wins > bm25_wins:
                print(f"🥇 VSM CHIẾN THẮNG TỔNG THỂ!")
            elif bm25_wins > vsm_wins:
                print(f"🥇 BM25 CHIẾN THẮNG TỔNG THỂ!")
            else:
                print(f"🤝 HÒA NHAU!")
        else:
            print("❌ Không thể trích xuất metrics để so sánh")
    else:
        print("❌ Không thể so sánh do một hoặc cả hai mô hình gặp lỗi")
    
    # Thời gian thực hiện
    print(f"\n⏱️  THỜI GIAN THỰC HIỆN:")
    if vsm_success:
        print(f"VSM:  {vsm_time:.2f} giây")
    else:
        print(f"VSM:  Lỗi")
        
    if bm25_success:
        print(f"BM25: {bm25_time:.2f} giây")
    else:
        print(f"BM25: Lỗi")
    
    if vsm_success and bm25_success:
        total_time = vsm_time + bm25_time
        print(f"Tổng: {total_time:.2f} giây")
        
        if vsm_time < bm25_time:
            print(f"🚀 VSM nhanh hơn {bm25_time - vsm_time:.2f} giây")
        elif bm25_time < vsm_time:
            print(f"🚀 BM25 nhanh hơn {vsm_time - bm25_time:.2f} giây")
        else:
            print("⚡ Cả hai cùng tốc độ")

if __name__ == "__main__":
    main() 