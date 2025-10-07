import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==========================================================
# Modul preprocessing survei mahasiswa — versi ringan
# ==========================================================

# -----------------------------------------------
# ============== PREPROCESSING UTILS =============
# -----------------------------------------------

def normalize_colnames(cols):
    """Rapikan nama kolom: hilangkan spasi berlebih dan trim."""
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]


def normalize_text_series(s: pd.Series) -> pd.Series:
    """Normalisasi string: lowercase, trim, samakan en-dash -> hyphen, rapikan spasi."""
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace("\u2013", "-", regex=False)  # en-dash -> hyphen
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def parse_range_midpoint(s: pd.Series) -> pd.Series:
    """Parse rentang 'x - y' menjadi midpoint (float). Contoh: '3 - 5 jam' -> 4.0"""
    s_str = s.astype(str)
    m = s_str.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    has_range = m[0].notna() & m[1].notna()
    midpoint = m[has_range].astype(float).mean(axis=1)
    out = s.copy()
    out.loc[has_range] = midpoint
    return out


# -----------------------------------------------
# ============ PREPROCESSING FUNCTIONS ===========
# -----------------------------------------------

def manual_preprocess_baseline(data: pd.DataFrame):
    """
    Fungsi preprocessing baseline (untuk digunakan jika diminta)
    - Drop kolom yang tidak relevan
    - Normalisasi kolom numerik, dan lainnya
    """
    df = data.copy()

    # Misalnya drop kolom yang tidak diperlukan
    drop_cols = ["timestamp", "date", "email", "nama", "name", "semester"]
    df = df.drop(columns=[col for col in df.columns if col in drop_cols])

    # Normalisasi kolom string
    df.columns = normalize_colnames(df.columns)
    for c in df.select_dtypes("object").columns:
        df[c] = normalize_text_series(df[c])

    # Imputasi untuk nilai yang hilang
    df = df.fillna(df.median())

    return df

def manual_preprocess_v2(
    data: pd.DataFrame,
    drop_cols=("timestamp", "date", "email", "nama", "name", "semester"),
    outlier="clip",          # "clip" (winsorize by IQR) atau "drop"
    scale="robust",          # "standard" | "robust" | None
    small_cat_max_card=12,   # one-hot untuk kategori kecil
    skew_thresh=1.0
):
    """
    PREPROCESSING IMPROVED v2:
    - Drop kolom identitas umum
    - Normalisasi teks & mapping ordinal/boolean
    - One-hot kategori kecil
    - Imputasi median
    - Outlier handling (clip/winsorize per kolom)
    - Log-transform bila skew > threshold
    - Scaling robust / standard / none
    """
    df = data.copy()

    # 1) Normalisasi header & string
    df.columns = normalize_colnames(df.columns)
    for c in df.select_dtypes("object").columns:
        df[c] = normalize_text_series(df[c])

    # 2) Drop kolom identitas/timestamp umum
    to_drop = [c for c in df.columns if any(k in c.lower() for k in drop_cols)]
    if to_drop:
        df = df.drop(columns=to_drop)

    # 3) Mapping nilai ordinal, boolean, dan rentang
    base_map = {
        "kurang dari 1 jam": 0.5,
        "1 - 2 jam": 1.5, "1-2 jam": 1.5,
        "3 - 5 jam": 4.0, "3-5 jam": 4.0,
        "6 - 8 jam": 7.0, "6-8 jam": 7.0,
        "9 - 12 jam": 10.5, "9-12 jam": 10.5,
        "lebih dari 12 jam": 12.5,
        "selalu": 4, "sering": 3, "kadang-kadang": 2, "jarang": 1, "tidak pernah": 0,
        "jarang (1 - 2 kali per semester)": 1,
        "kadang-kadang (1 - 2 kali per bulan)": 2,
        "sering (hampir tiap bulan)": 3,
        "sangat sering (lebih dari 1 kali per bulan)": 4,
        "sangat penting": 4, "penting": 3, "cukup penting": 2, "tidak terlalu penting": 1,
        "ya": 1, "tidak": 0,
        "sangat seimbang": 4, "seimbang": 3, "kurang seimbang": 2, "tidak seimbang": 1,
    }

    obj_cols = df.select_dtypes("object").columns.tolist()
    for c in obj_cols:
        s = parse_range_midpoint(df[c])
        s = s.replace(base_map)
        s_num = pd.to_numeric(s, errors="coerce")
        df[c] = np.where(s_num.notna(), s_num, s)

    # 4) One-hot untuk kategori kecil
    obj_cols_after = df.select_dtypes("object").columns.tolist()
    small_cats = []
    for c in obj_cols_after:
        uniq = df[c].dropna().unique()
        if 1 < len(uniq) <= small_cat_max_card:
            small_cats.append(c)
    if small_cats:
        df = pd.get_dummies(df, columns=small_cats, drop_first=False, dtype=float)

    # 5) Ambil numerik + imputasi median
    df_num = df.select_dtypes(include=["number"]).copy()
    df_num = df_num.fillna(df_num.median(numeric_only=True))

    # 6) Outlier handling
    Q1, Q3 = df_num.quantile(0.25), df_num.quantile(0.75)
    IQR = (Q3 - Q1).replace(0, np.nan)
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    if outlier == "clip":
        df_num = df_num.clip(lower=lower, upper=upper, axis=1)
    elif outlier == "drop":
        mask = ~((df_num.lt(lower)) | (df_num.gt(upper))).any(axis=1)
        df_num = df_num[mask]

    # 7) Log-transform jika miring & non-negatif
    for c in df_num.columns:
        col = df_num[c]
        if col.min() >= 0:
            sk = col.skew()
            if np.isfinite(sk) and abs(sk) > skew_thresh:
                df_num[c] = np.log1p(col)

    # 8) Scaling
    if scale == "standard":
        mean = df_num.mean()
        std = df_num.std(ddof=0).replace(0, 1.0)
        X_scaled = (df_num - mean) / std
    elif scale == "robust":
        med = df_num.median()
        iqr = (df_num.quantile(0.75) - df_num.quantile(0.25)).replace(0, 1.0)
        X_scaled = (df_num - med) / iqr
    else:
        X_scaled = df_num.copy()

    # 9) Laporan ringkas
    report = {
        "mode": "improved_v2",
        "dropped_cols": to_drop,
        "onehot_cols": small_cats,
        "n_rows": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "outlier_strategy": outlier,
        "scale": scale,
    }

    return df_num, X_scaled, report

# ==========================================================
# KMeans Manual Implementation
# ==========================================================

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

    def visualize_clustering_process(self, X, labels, centroids, feature_names):
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


# ==========================================================
# PREPROCESSING UTILS (data_cleaning functions)
# ==========================================================

def normalize_colnames(cols):
    """Rapikan nama kolom: hilangkan spasi berlebih dan trim."""
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]


def normalize_text_series(s: pd.Series) -> pd.Series:
    """Normalisasi string: lowercase, trim, samakan en-dash -> hyphen, rapikan spasi."""
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace("\u2013", "-", regex=False)  # en-dash -> hyphen
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def parse_range_midpoint(s: pd.Series) -> pd.Series:
    """Parse rentang 'x - y' menjadi midpoint (float). Contoh: '3 - 5 jam' -> 4.0"""
    s_str = s.astype(str)
    m = s_str.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    has_range = m[0].notna() & m[1].notna()
    midpoint = m[has_range].astype(float).mean(axis=1)
    out = s.copy()
    out.loc[has_range] = midpoint
    return out

# ===================== DBSCAN MANUAL =====================

def gower_distance(df):
    """
    Menghitung matriks jarak Gower untuk dataset dengan data campuran (numerik dan kategorikal).
    :param df: Dataframe yang berisi data campuran (numerik dan kategorikal).
    :return: Matriks jarak Gower
    """
    n_samples = df.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            distance = 0
            for col in df.columns:
                if df[col].dtype == 'object':  # Untuk kolom kategori
                    distance += (df.iloc[i][col] != df.iloc[j][col])
                else:  # Untuk kolom numerik
                    distance += abs(df.iloc[i][col] - df.iloc[j][col]) / (df[col].max() - df[col].min())
            distance_matrix[i, j] = distance / len(df.columns)

    return distance_matrix

def calculate_distance_matrix(X):
    """Menghitung jarak antara semua pasangan titik data"""
    return np.linalg.norm(X[:, np.newaxis] - X, axis=2)

def dbscan(X, eps=0.5, min_samples=5):
    """
    Implementasi manual DBSCAN
    :param X: Data (n_samples, n_features)
    :param eps: radius tetangga untuk dua titik agar dianggap saling terhubung
    :param min_samples: jumlah minimal tetangga yang diperlukan untuk membentuk cluster
    :return: labels: array yang berisi label cluster untuk setiap titik (noise akan mendapat label -1)
    """
    n_samples = X.shape[0]
    distance_matrix = calculate_distance_matrix(X)
    labels = -1 * np.ones(n_samples)  # Semua titik diawali dengan label noise (-1)
    cluster_id = 0  # ID cluster dimulai dari 0

    def region_query(point_idx):
        """Menemukan tetangga dalam radius eps"""
        return np.where(distance_matrix[point_idx] <= eps)[0]

    def expand_cluster(point_idx, neighbors, cluster_id):
        """Mengekspansi cluster dengan menambahkan titik-titik yang terhubung"""
        labels[point_idx] = cluster_id  # Tandai titik sebagai bagian dari cluster
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # Jika titik tersebut adalah noise, ubah menjadi anggota cluster
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                # Jika titik tetangga memiliki cukup banyak tetangga, ekspansi cluster
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1

    for point_idx in range(n_samples):
        if labels[point_idx] != -1:
            continue  # Jika titik sudah termasuk dalam cluster, lanjutkan
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Jika tidak cukup tetangga, tandai sebagai noise
        else:
            expand_cluster(point_idx, neighbors, cluster_id)
            cluster_id += 1

    return labels

def euclidean_distance_matrix(X):
    """Hitung matriks jarak Euclidean manual antar semua titik."""
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((X[i] - X[j])**2))
            D[i, j] = dist
            D[j, i] = dist
    return D

def plot_k_distance_graph_manual(X, k=5):
    """Plot k-distance graph tanpa sklearn."""
    D = euclidean_distance_matrix(X)
    D_sorted = np.sort(D, axis=1)
    k_distances = D_sorted[:, k]
    k_distances_sorted = np.sort(k_distances)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(np.arange(len(k_distances_sorted)), k_distances_sorted)
    ax.set_title(f"{k}-distance Plot (manual, cari eps ≈ titik siku)")
    ax.set_xlabel("Data point (sorted)")
    ax.set_ylabel(f"Distance ke-{k}")
    ax.grid(True)
    
    return fig, k_distances_sorted  # Return figure dan array

def pca_manual(X, n_components=2):
    """Lakukan PCA manual (tanpa sklearn)"""
    # 1️⃣ Pusatkan data (zero-mean)
    X_centered = X - X.mean(axis=0)
    
    # 2️⃣ Hitung covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # 3️⃣ Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 4️⃣ Urutkan eigenvalue dari terbesar ke kecil
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    
    # 5️⃣ Ambil komponen utama (dua pertama)
    components = eigvecs[:, :n_components]
    X_reduced = np.dot(X_centered, components)
    return X_reduced

def visualize_dbscan_results(X, labels):
    """Visualisasi hasil DBSCAN dengan PCA 2D"""
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

# ===================== STREAMLIT APP =====================

def main():
    st.set_page_config(page_title="Preprocessing & Clustering", page_icon="🧹", layout="wide")
    
    st.title("🧹 Preprocessing & 🔗 Clustering")

    tab1, tab2 = st.tabs(["1) Preprocessing", "2) Clustering"])

    with tab1:
        st.sidebar.header("⚙️ Opsi Preprocessing")
        mode = st.sidebar.selectbox("Mode", ["Improved v2 (disarankan)", "Baseline (awal)"])
        uploaded_file = st.file_uploader("Unggah file CSV/XLSX (Preprocessing)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
            st.subheader("📂 Data Asli (preview)")
            st.dataframe(data.head(15))

            if st.button("🚀 Proses Data"):
                with st.spinner("Memproses..."):
                    if mode == "Improved v2 (disarankan)":
                        imputed, X_scaled, report = manual_preprocess_v2(data)
                    else:
                        imputed, X_scaled, report = manual_preprocess_baseline(data)

                st.success("✅ Selesai!")
                st.session_state["clean_imputed"] = imputed
                st.session_state["X_scaled"] = X_scaled

                st.subheader("🔎 Data Setelah Preprocessing")
                st.dataframe(imputed.head(25))

                st.subheader("📝 Report")
                st.json(report)

    with tab2:
        st.header("🔗 Clustering (KMeans & DBSCAN)")

        df_for_cluster = None
        src = st.radio("Pilih sumber data clustering", ["Gunakan hasil Preprocessing", "Upload file baru"], horizontal=True)
        
        if src == "Gunakan hasil Preprocessing":
            if "clean_imputed" in st.session_state:
                df_for_cluster = st.session_state["clean_imputed"].copy()
            else:
                st.warning("Silakan proses data di Tab 1 terlebih dahulu.")
        else:
            uploaded_file2 = st.file_uploader("Unggah file CSV/XLSX untuk clustering", type=["csv", "xlsx"])
            if uploaded_file2 is not None:
                df_for_cluster = pd.read_csv(uploaded_file2) if uploaded_file2.name.lower().endswith(".csv") else pd.read_excel(uploaded_file2)

        if df_for_cluster is not None:
            st.write("Preview data untuk clustering:")
            st.dataframe(df_for_cluster.head(10))

            method = st.selectbox("Metode", ["KMeans", "DBSCAN"])

            D = gower_distance(df_for_cluster)

            if method == "KMeans":
                k = st.number_input("Jumlah cluster (k)", min_value=2, max_value=30, value=4)
                if st.button("▶️ Jalankan KMeans"):
                    with st.spinner("Clustering..."):
                        kmeans = KMeans(k=k, max_iters=100, random_state=42)
                        kmeans.fit(D)
                        df_for_cluster["cluster"] = kmeans.labels
                        st.success("✅ KMeans selesai!")
                        st.dataframe(df_for_cluster.head(20))
                        st.subheader("📈 Visualisasi Clustering")
                        fig = kmeans.visualize_clustering_process(D, kmeans.labels, kmeans.centroids, df_for_cluster.columns)
                        st.pyplot(fig)

            elif method == "DBSCAN":
                st.subheader("📊 K-Distance Plot (untuk menentukan eps)")
                col1, col2 = st.columns([1, 3])
    
                with col1:
                    k_value = st.slider("Pilih nilai k", min_value=3, max_value=20, value=5, 
                                        help="Biasanya k = min_samples")
                    if st.button("🔍 Tampilkan K-Distance Plot"):
                        st.session_state['show_k_plot'] = True
    
                with col2:
                    if st.session_state.get('show_k_plot', False):
                        with st.spinner("Menghitung k-distance..."):
                            fig, k_dist = plot_k_distance_graph_manual(df_for_cluster.values, k=k_value)
                            st.pyplot(fig)
                            st.info(f"💡 Cari 'titik siku' (elbow) pada grafik. Nilai Y di titik siku adalah **eps optimal**")
                st.divider()
                # Parameter DBSCAN
                st.subheader("⚙️ Parameter DBSCAN")
                eps = st.number_input("eps", min_value=0.01, max_value=5.0, value=0.35, 
                                    help="Radius maksimum untuk mencari tetangga")
                min_samples = st.number_input("min_samples", min_value=3, max_value=50, value=6,
                                    help="Jumlah minimum tetangga untuk membentuk cluster")
    
                if st.button("▶️ Jalankan DBSCAN"):
                    with st.spinner("Clustering..."):
                        db_labels = dbscan(df_for_cluster.values, eps=eps, min_samples=min_samples)
            
                        # Hitung statistik cluster
                        unique_labels = np.unique(db_labels)
                        n_clusters = len(unique_labels[unique_labels != -1])
                        n_noise = np.sum(db_labels == -1)
            
                        df_for_cluster["cluster"] = db_labels
                        st.success(f"✅ DBSCAN selesai! **{n_clusters} cluster** ditemukan, **{n_noise} noise points**")
                        
                        # Tampilkan distribusi cluster
                        st.write("**Distribusi Cluster:**")
                        cluster_counts = pd.Series(db_labels).value_counts().sort_index()
                        st.dataframe(cluster_counts.to_frame("Jumlah"))
                        
                        st.dataframe(df_for_cluster.head(20))
                        
                        st.subheader("📈 Visualisasi DBSCAN")
                        fig_dbscan = visualize_dbscan_results(df_for_cluster.values, db_labels)
                        st.pyplot(fig_dbscan)


            else:
                st.info("Pilih sumber data dan/atau upload file untuk clustering.")
    
if __name__ == "__main__":
    main()
