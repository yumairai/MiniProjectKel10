import numpy as np
import matplotlib.pyplot as plt

# ======== Distance helpers ========

def euclidean_distance_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            D[i, j] = dist
            D[j, i] = dist
    return D

def calculate_distance_matrix(X):
    return np.linalg.norm(X[:, np.newaxis] - X, axis=2)

# ======== DBSCAN (manual) ========

def dbscan(X, eps=0.5, min_samples=5):
    """
    0 = unvisited, -1 = noise, >=1 = cluster id
    """
    n_samples = X.shape[0]
    D = calculate_distance_matrix(X)
    labels = np.zeros(n_samples, dtype=int)  # 0 = unvisited
    cluster_id = 0

    def neighbors(i):
        return np.where(D[i] <= eps)[0]

    for i in range(n_samples):
        if labels[i] != 0:
            continue
        Ni = neighbors(i)
        if len(Ni) < min_samples:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(Ni[Ni != i])
        idx = 0
        while idx < len(seeds):
            j = seeds[idx]
            if labels[j] == -1:
                labels[j] = cluster_id  # upgrade noise -> border
            if labels[j] == 0:
                labels[j] = cluster_id
                Nj = neighbors(j)
                if len(Nj) >= min_samples:
                    for p in Nj:
                        if p not in seeds:
                            seeds.append(p)
            idx += 1

    return labels

# ======== K-Distance plot & PCA viz ========

def plot_k_distance_graph_manual(X, k=5):
    D = euclidean_distance_matrix(X)
    D_sorted = np.sort(D, axis=1)
    k_distances = D_sorted[:, k]
    k_distances_sorted = np.sort(k_distances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(k_distances_sorted)), k_distances_sorted)
    ax.set_title(f"{k}-distance Plot (manual, cari eps ≈ titik siku)")
    ax.set_xlabel("Data point (sorted)")
    ax.set_ylabel(f"Distance ke-{k}")
    ax.grid(True)
    return fig, k_distances_sorted

def pca_manual(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx][:, :n_components]
    X_reduced = np.dot(X_centered, components)
    return X_reduced

def visualize_dbscan_results(X, labels):
    X_2d = pca_manual(X, n_components=2)
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c='red', s=60,
                       label='Noise (-1)', alpha=0.7, edgecolors='k')
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=60,
                       label=f'Cluster {lbl}', alpha=0.7)
    ax.set_title("DBSCAN Clustering Results (PCA 2D Projection)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    ax.grid(True)
    return fig
