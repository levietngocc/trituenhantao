# File: src/clustering.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import logging
from src.visualize import plot_elbow_and_silhouette, plot_dbscan_kdistance, plot_gmm_criteria

logger = logging.getLogger('PipelineLogger')

def kmeans_clustering(X, max_k=10, random_state=42, save_dir="outputs/figures"):
    logger.info("Bắt đầu phân cụm bằng KMeans...")
    inertias = []
    sil_scores = []
    k_range = range(2, max_k + 1)
    
    models = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        models[k] = kmeans
        
    plot_elbow_and_silhouette(inertias, sil_scores, k_range, save_path=f"{save_dir}/kmeans_elbow.png")
    
    best_k = k_range[np.argmax(sil_scores)]
    logger.info(f"KMeans: Chọn k tối ưu = {best_k} dựa trên Silhouette Score.")
    
    best_model = models[best_k]
    return {
        'labels': best_model.labels_,
        'model': best_model,
        'n_clusters': best_k,
        'scores': {'inertias': inertias, 'silhouettes': sil_scores}
    }

def dbscan_clustering(X, eps_range=None, min_samples=5, save_dir="outputs/figures"):
    logger.info("Bắt đầu phân cụm bằng DBSCAN...")
    
    # K-distance graph để tìm eps
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    plot_dbscan_kdistance(distances, min_samples, save_path=f"{save_dir}/dbscan_kdistance.png")
    
    # Simple heuristic to find knee: point of maximum curvature / highest gradient change
    gradients = np.diff(distances)
    knee_idx = np.argmax(gradients) + 1
    best_eps = distances[knee_idx]
    
    # Hardcode a fallback if heuristic fails (too small or too large)
    if best_eps < 0.1 or best_eps > 2.0:
        best_eps = 0.5 
        
    logger.info(f"DBSCAN: Eps tự động ước lượng = {best_eps:.3f}")
    
    dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)
    
    logger.info(f"DBSCAN: Tìm thấy {n_clusters} cụm, {n_noise} điểm nhiễu ({noise_ratio:.1%}).")
    if noise_ratio > 0.5:
        logger.warning("DBSCAN: Tỷ lệ noise > 50%. Tham số eps có thể chưa phù hợp.")
        
    return {
        'labels': labels,
        'model': dbscan,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

def gmm_clustering(X, max_components=10, criterion='bic', random_state=42, save_dir="outputs/figures"):
    logger.info("Bắt đầu phân cụm bằng GMM...")
    bics = []
    aics = []
    comp_range = range(2, max_components + 1)
    
    models = {}
    for n in comp_range:
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        models[n] = gmm
        
    plot_gmm_criteria(comp_range, bics, 'BIC', save_path=f"{save_dir}/gmm_bic.png")
    plot_gmm_criteria(comp_range, aics, 'AIC', save_path=f"{save_dir}/gmm_aic.png")
    
    if criterion == 'bic':
        best_n = comp_range[np.argmin(bics)]
    else:
        best_n = comp_range[np.argmin(aics)]
        
    logger.info(f"GMM: Chọn số component tối ưu = {best_n} dựa trên {criterion.upper()}.")
    
    best_model = models[best_n]
    labels = best_model.predict(X)
    
    return {
        'labels': labels,
        'model': best_model,
        'n_clusters': best_n,
        'bic_scores': bics
    }