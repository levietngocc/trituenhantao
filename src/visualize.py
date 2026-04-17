# File: src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.utils import ensure_dir

def setup_plot_style():
    sns.set_theme(style="whitegrid", palette="muted")

def plot_elbow_and_silhouette(inertias, silhouette_scores, k_range, save_path):
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Số lượng cụm (k)')
    ax1.set_ylabel('Inertia', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('KMeans: Biểu đồ Elbow và Silhouette')
    fig.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_dbscan_kdistance(distances, k, save_path):
    setup_plot_style()
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'DBSCAN: K-Distance Graph (k={k})')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-NN Distance')
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_gmm_criteria(components, scores, criterion_name, save_path):
    setup_plot_style()
    plt.figure(figsize=(10, 6))
    plt.plot(components, scores, marker='o')
    plt.title(f'GMM: {criterion_name} Scores theo số lượng cụm')
    plt.xlabel('Số lượng components')
    plt.ylabel(f'{criterion_name} Score')
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_clusters_2d(X_pca, labels, title, save_path):
    setup_plot_style()
    plt.figure(figsize=(10, 8))
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1] # Black for noise
            
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask]
        
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}',
                    edgecolors='k', s=50, alpha=0.7)
        
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cluster_profiles_heatmap(profiles_df, save_path):
    setup_plot_style()
    plt.figure(figsize=(12, 8))
    
    # Scale profile để vẽ heatmap dễ nhìn
    scaler = sns.clustermap(profiles_df, cmap='YlGnBu', annot=True, fmt=".2f", standard_scale=1, figsize=(10, 8))
    
    ensure_dir(os.path.dirname(save_path))
    scaler.savefig(save_path, dpi=300)
    plt.close()