# File: src/ablation.py
import pandas as pd
import logging
from src.preprocess import Preprocessor
from src.clustering import kmeans_clustering, dbscan_clustering, gmm_clustering
from src.evaluate import compare_clustering_methods

logger = logging.getLogger('PipelineLogger')

def run_ablation(raw_df):
    logger.info("=== BẮT ĐẦU ABLATION STUDY ===")
    preprocessor = Preprocessor()
    cleaned_df = preprocessor.load_and_clean(raw_df)
    rfm_raw = preprocessor.build_rfm_features(cleaned_df)
    
    # Baseline (Log + Scale + PCA)
    rfm_log = preprocessor.transform_features(rfm_raw)
    X_scaled = preprocessor.scale_features(rfm_log)
    X_pca = preprocessor.apply_pca(X_scaled)
    
    # Exp 1: Bỏ Log Transform (Chỉ Scale nguyên bản)
    X_exp1 = preprocessor.scale_features(rfm_raw)
    
    # Exp 2: Bỏ PCA (Sử dụng toàn bộ không gian đặc trưng đã log & scale)
    X_exp2 = X_scaled 
    
    experiments = {
        'Baseline (Log+Scale+PCA)': X_pca,
        'Bỏ Log (Raw+Scale)': X_exp1,
        'Bỏ PCA (Log+Scale)': X_exp2
    }
    
    all_results = []
    
    for exp_name, X in experiments.items():
        logger.info(f"Chạy thí nghiệm: {exp_name}")
        # Chạy thuật toán với tham số mặc định nhanh
        km = kmeans_clustering(X, max_k=5)['labels']
        gm = gmm_clustering(X, max_components=5)['labels']
        
        methods_dict = {'KMeans': km, 'GMM': gm}
        res_df = compare_clustering_methods(X, methods_dict)
        res_df['Experiment'] = exp_name
        all_results.append(res_df)
        
    final_res = pd.concat(all_results, ignore_index=True)
    logger.info("=== KẾT THÚC ABLATION STUDY ===")
    return final_res