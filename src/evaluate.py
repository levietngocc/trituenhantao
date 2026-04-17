# File: src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import logging

logger = logging.getLogger('PipelineLogger')

def silhouette_score_robust(X, labels, noise_label=-1):
    mask = labels != noise_label
    if len(set(labels[mask])) >= 2:
        return silhouette_score(X[mask], labels[mask])
    return np.nan

def davies_bouldin_score_robust(X, labels, noise_label=-1):
    mask = labels != noise_label
    if len(set(labels[mask])) >= 2:
        return davies_bouldin_score(X[mask], labels[mask])
    return np.nan

def calinski_harabasz_score_robust(X, labels, noise_label=-1):
    mask = labels != noise_label
    if len(set(labels[mask])) >= 2:
        return calinski_harabasz_score(X[mask], labels[mask])
    return np.nan

def bootstrap_stability(X, clustering_func, n_iter=10, sample_frac=0.8, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    sample_size = int(n_samples * sample_frac)
    
    # Lần chạy chuẩn để so sánh
    base_labels = clustering_func(X)
    ari_scores = []
    
    for i in range(n_iter):
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[indices]
        sample_labels = clustering_func(X_sample)
        
        # Chỉ so sánh ARI trên các điểm dữ liệu chung
        ari = adjusted_rand_score(base_labels[indices], sample_labels)
        ari_scores.append(ari)
        
    return np.mean(ari_scores), np.std(ari_scores)

def compare_clustering_methods(X, methods_dict):
    logger.info("Đang đánh giá và so sánh các thuật toán...")
    results = []
    
    for name, labels in methods_dict.items():
        sil = silhouette_score_robust(X, labels)
        dbi = davies_bouldin_score_robust(X, labels)
        ch = calinski_harabasz_score_robust(X, labels)
        
        # Đếm noise nếu có
        n_noise = list(labels).count(-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        results.append({
            'Method': name,
            'Clusters': n_clusters,
            'Noise points': n_noise,
            'Silhouette (↑)': round(sil, 4) if not np.isnan(sil) else None,
            'Davies-Bouldin (↓)': round(dbi, 4) if not np.isnan(dbi) else None,
            'Calinski-Harabasz (↑)': round(ch, 2) if not np.isnan(ch) else None
        })
        
    return pd.DataFrame(results)