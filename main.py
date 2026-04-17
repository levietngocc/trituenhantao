# File: main.py
import os
from src.utils import set_seed, setup_logger, load_data, save_dataframe
from src.preprocess import Preprocessor
from src.clustering import kmeans_clustering, dbscan_clustering, gmm_clustering
from src.evaluate import compare_clustering_methods
from src.visualize import plot_clusters_2d, plot_cluster_profiles_heatmap
import pandas as pd

def main():
    # 1. Setup
    set_seed(42)
    logger = setup_logger('PipelineLogger', 'outputs/logs/pipeline.log')
    logger.info("=== KHỞI CHẠY PIPELINE PHÂN CỤM KHÁCH HÀNG ===")
    
    # 2. Load data
    data_path = 'data/data.csv'
    try:
        raw_df = load_data(data_path)
    except FileNotFoundError:
        print("Vui lòng tạo thư mục 'data/' và đặt file data.csv vào đó.")
        return

    # 3. Preprocess
    preprocessor = Preprocessor()
    processed_data = preprocessor.get_processed_data(raw_df)
    
    X_scaled = processed_data['X_scaled']
    X_pca = processed_data['X_pca']
    rfm_raw = processed_data['rfm_raw']
    customer_ids = processed_data['customer_ids']
    
    # 4. Clustering (trên không gian PCA cho nhẹ và trực quan)
    logger.info("--- Thực hiện chạy các thuật toán ---")
    km_res = kmeans_clustering(X_pca)
    db_res = dbscan_clustering(X_pca)
    gmm_res = gmm_clustering(X_pca)
    
    methods_dict = {
        'KMeans': km_res['labels'],
        'DBSCAN': db_res['labels'],
        'GMM': gmm_res['labels']
    }
    
    # 5. Đánh giá
    evaluation_df = compare_clustering_methods(X_pca, methods_dict)
    print("\n--- BẢNG SO SÁNH KẾT QUẢ ---")
    print(evaluation_df.to_markdown(index=False))
    save_dataframe(evaluation_df, 'outputs/tables/evaluation_metrics.csv')
    
    # 6. Trực quan hóa không gian phân cụm 2D
    plot_clusters_2d(X_pca, km_res['labels'], "Phân cụm bằng KMeans", "outputs/figures/scatter_kmeans.png")
    plot_clusters_2d(X_pca, db_res['labels'], "Phân cụm bằng DBSCAN", "outputs/figures/scatter_dbscan.png")
    plot_clusters_2d(X_pca, gmm_res['labels'], "Phân cụm bằng GMM", "outputs/figures/scatter_gmm.png")
    
    # 7. Profiling với GMM (Giả định GMM cho kết quả linh hoạt nhất)
    logger.info("--- Tạo Profile các cụm (Dùng kết quả GMM) ---")
    rfm_raw['Cluster'] = gmm_res['labels']
    
    # Tính trung bình các đặc trưng theo cụm
    profile_df = rfm_raw.groupby('Cluster').mean().round(2)
    profile_counts = rfm_raw['Cluster'].value_counts().rename("CustomerCount")
    profile_df = profile_df.join(profile_counts)
    
    print("\n--- ĐẶC TRƯNG CÁC CỤM ---")
    print(profile_df.to_markdown())
    save_dataframe(profile_df.reset_index(), 'outputs/tables/cluster_profiles.csv')
    
    # Vẽ heatmap profile
    plot_cluster_profiles_heatmap(profile_df.drop('CustomerCount', axis=1), 'outputs/figures/profile_heatmap.png')
    
    # Gợi ý Marketing
    logger.info("Đã xuất đặc trưng cụm. Dựa vào bảng trên để đưa ra chính sách Marketing phù hợp.")
    
    # 8. Lưu nhãn
    final_labels = pd.DataFrame({'CustomerID': customer_ids, 'Cluster': gmm_res['labels']})
    save_dataframe(final_labels, 'outputs/tables/customer_labels.csv')
    
    logger.info("=== PIPELINE HOÀN TẤT THÀNH CÔNG ===")

if __name__ == "__main__":
    main()