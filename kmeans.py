import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class KMeans:
    def __init__(self, k=4, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
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


def describe_cluster(centroid, feature_names):
    """Membuat deskripsi otomatis berdasarkan nilai centroid"""
    desc_parts = []
    for f, val in zip(feature_names, centroid):
        if val >= 0.7:
            desc_parts.append(f"tinggi pada **{f}**")
        elif val <= 0.3:
            desc_parts.append(f"rendah pada **{f}**")
        else:
            desc_parts.append(f"sedang pada **{f}**")
    return ", ".join(desc_parts)


def generate_cluster_insights_auto(df, kmeans, feature_names):
    """Analisis otomatis tiap cluster"""
    insights = {}
    for i in range(kmeans.k):
        cluster_data = df[df['cluster'] == i].drop('cluster', axis=1)
        mean_values = cluster_data.mean()
        desc = describe_cluster(mean_values.values, feature_names)
        insights[i] = {
            "description": f"Cluster ini memiliki nilai {desc}.",
            "mean": mean_values
        }
    return insights

def visualize_clustering_process(X, labels, centroids, feature_names):
    """
    Membuat visualisasi proses clustering dengan PCA manual (tanpa sklearn)
    """
    # --- PCA manual ---
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1e-8
    X_scaled = (X - X_mean) / X_std
    cov_matrix = np.cov(X_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    components = eigenvectors[:, :2]
    X_pca = np.dot(X_scaled, components)
    centroids_scaled = (centroids - X_mean) / X_std
    centroids_pca = np.dot(centroids_scaled, components)

    # --- Visualisasi ---
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", np.unique(labels).size)
    for i in range(np.unique(labels).size):
        cluster_points = X_pca[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   label=f"Cluster {i}",
                   color=palette[i], edgecolor='black', s=60, alpha=0.7)
    
    # Plot centroid
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               color='black', marker='X', s=200, label='Centroid')

    ax.set_xlabel("Komponen Utama 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Komponen Utama 2", fontsize=12, fontweight="bold")
    ax.set_title("Visualisasi Proses Clustering (PCA 2D)", fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig



# ===================== STREAMLIT APP =====================

def main():
    st.set_page_config(page_title="K-Means Clustering Mahasiswa", page_icon="🎓", layout="wide")
    
    st.title("🎓 Eksplorasi Clustering Aktivitas Mahasiswa")
    st.markdown("Kita akan mencoba menemukan **4 kelompok (cluster)** mahasiswa berdasarkan pola aktivitas mereka — tanpa asumsi awal tentang makna tiap cluster.")
    st.markdown("---")

    uploaded_file = st.sidebar.file_uploader("📁 Upload file CSV hasil preprocessing", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} fitur")

        st.write("**Contoh Data:**")
        st.dataframe(df.head())

        X = df.values
        feature_names = df.columns.tolist()

        with st.spinner("🔄 Melakukan clustering..."):
            kmeans = KMeans(k=4, max_iters=100, random_state=42)
            kmeans.fit(X)
            df['cluster'] = kmeans.labels

        st.success("✅ Clustering selesai!")

        st.header("📊 Hasil Clustering")
        inertia = kmeans.calculate_inertia(X)
        st.metric("Inertia (WCSS)", f"{inertia:.2f}")

        cluster_counts = df['cluster'].value_counts().sort_index()
        cluster_pct = (cluster_counts / len(df) * 100).round(2)

        col1, col2 = st.columns([1, 2])
        with col1:
            summary = pd.DataFrame({
                "Jumlah Mahasiswa": cluster_counts,
                "Persentase": cluster_pct
            })
            st.dataframe(summary)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette("Set2", 4))
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"Cluster {i}" for i in range(4)])
            ax.set_ylabel("Jumlah Mahasiswa")
            ax.set_title("Distribusi Tiap Cluster")
            st.pyplot(fig)

        # Analisis otomatis
        st.header("🔍 Interpretasi Otomatis Tiap Cluster")
        insights = generate_cluster_insights_auto(df, kmeans, feature_names)

        for i in range(4):
            with st.expander(f"Cluster {i}", expanded=True):
                st.markdown(f"**Deskripsi:** {insights[i]['description']}")
                st.write("**Rata-rata nilai per fitur:**")
                st.dataframe(insights[i]['mean'].round(2))

        # Heatmap
        st.markdown("---")
        st.subheader("🔥 Visualisasi Centroid (Heatmap)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(kmeans.centroids.T, annot=True, cmap="YlGnBu", 
                    xticklabels=[f"Cluster {i}" for i in range(4)],
                    yticklabels=feature_names, ax=ax)
        st.pyplot(fig)

        # Kesimpulan
        st.markdown("---")
        st.header("📌 Kesimpulan")
        st.markdown(f"""
        Dari hasil clustering, ditemukan **4 kelompok mahasiswa** dengan karakteristik berbeda.
        Setiap cluster menggambarkan pola kecenderungan aktivitas berdasarkan fitur seperti:
        **{', '.join(feature_names)}.**

        Gunakan hasil ini untuk mengenali pola umum mahasiswa — misalnya kelompok yang cenderung akademis, 
        aktif berorganisasi, atau memiliki keseimbangan aktivitas yang berbeda.
        """)

        # Download hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Hasil Clustering (CSV)",
            csv,
            "hasil_clustering_eksploratif.csv",
            "text/csv"
        )
    else:
        st.info("👆 Upload file CSV untuk mulai analisis.")


if __name__ == "__main__":
    main()
