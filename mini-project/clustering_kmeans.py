import numpy as np

class KMeans:
    def __init__(self, k=4, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, _ = X.shape
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X, labels)
            if np.allclose(old_centroids, self.centroids):
                break

        self.labels = labels
        return self

    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                centroids[i] = X[np.random.choice(X.shape[0])]
        return centroids

    def calculate_inertia(self, X):
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia

def visualize_clustering_process(X, labels, centroids):
    # PCA manual untuk 2D
    import numpy as np
    import matplotlib.pyplot as plt

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1e-8
    X_scaled = (X - X_mean) / X_std
    cov_matrix = np.cov(X_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, sorted_idx][:, :2]
    X_pca = np.dot(X_scaled, components)
    centroids_scaled = (centroids - X_mean) / X_std
    centroids_pca = np.dot(centroids_scaled, components)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(np.unique(labels).size):
        cluster_points = X_pca[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], color='black', marker='X', s=200, label='Centroid')
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    ax.set_title("Visualisasi Proses Clustering (PCA 2D)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def elbow_method(X, max_k=10, random_state=42):
    inertia = []
    km = KMeans(random_state=random_state)
    for k in range(1, max_k + 1):
        km.k = k
        km.fit(X)
        inertia.append(km.calculate_inertia(X))
    return inertia

def gower_distance(df):
    import numpy as np
    n_samples = df.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance = 0
            for col in df.columns:
                if df[col].dtype == 'object':
                    distance += (df.iloc[i][col] != df.iloc[j][col])
                else:
                    rng = (df[col].max() - df[col].min())
                    rng = rng if rng != 0 else 1.0
                    distance += abs(df.iloc[i][col] - df.iloc[j][col]) / rng
            distance_matrix[i, j] = distance / len(df.columns)
    return distance_matrix
