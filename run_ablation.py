# File: run_ablation.py
from src.utils import set_seed, setup_logger, load_data, save_dataframe
from src.ablation import run_ablation

def main():
    set_seed(42)
    setup_logger('PipelineLogger', 'outputs/logs/ablation.log')
    
    try:
        raw_df = load_data('data/data.csv')
    except Exception as e:
        print(f"Lỗi: {e}")
        return
        
    ablation_results = run_ablation(raw_df)
    
    print("\n--- KẾT QUẢ ABLATION STUDY ---")
    # Sắp xếp để dễ so sánh
    ablation_results = ablation_results.sort_values(by=['Experiment', 'Method'])
    print(ablation_results.to_markdown(index=False))
    save_dataframe(ablation_results, 'outputs/tables/ablation_results.csv')

if __name__ == "__main__":
    main()