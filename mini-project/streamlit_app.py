import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
    if scale == "minmax":
        df_num_minmax = (df_num - df_num.min()) / (df_num.max() - df_num.min())
        X_scaled = df_num_minmax
    elif scale == "standard":
        mean = df_num.mean()
        std = df_num.std(ddof=0).replace(0, 1.0)
        X_scaled = (df_num - mean) / std
    elif scale == "robust":
        med = df_num.median()
        iqr = (df_num.quantile(0.75) - df_num.quantile(0.25)).replace(0, 1.0)
        X_scaled = (df_num - med) / iqr
    else:
        X_scaled = df_num.copy()

    # 9) Report
    report = {
        "mode": "improved_v2",
        "dropped_cols": to_drop,
        "onehot_cols": small_cats,
        "n_rows": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "outlier_strategy": outlier,
        "scale": scale,
    }

    return df_num, X_scaled, report  # Return 3 values


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
        for i in range(np.unique(labels).size):
            cluster_points = X_pca[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            color='black', marker='X', s=200, label='Centroid')

        ax.set_xlabel("Komponen Utama 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("Komponen Utama 2", fontsize=12, fontweight="bold")
        ax.set_title("Visualisasi Proses Clustering (PCA 2D)", fontsize=14, fontweight="bold", pad=15)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    def elbow_method(self, X, max_k=10):
        inertia = []
        for k in range(1, max_k + 1):
            self.k = k
            self.fit(X)
            inertia.append(self.calculate_inertia(X))
        return inertia


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
                if df[col].dtype == 'object':
                    distance += (df.iloc[i][col] != df.iloc[j][col])
                else:
                    distance += abs(df.iloc[i][col] - df.iloc[j][col]) / (df[col].max() - df[col].min())
            distance_matrix[i, j] = distance / len(df.columns)

    return distance_matrix


def calculate_distance_matrix(X):
    """Menghitung jarak antara semua pasangan titik data"""
    return np.linalg.norm(X[:, np.newaxis] - X, axis=2)


def dbscan(X, eps=0.5, min_samples=5):
    """
    Implementasi DBSCAN yang benar:
    0 = unvisited, -1 = noise, >=1 = cluster id
    """
    n_samples = X.shape[0]
    D = calculate_distance_matrix(X)
    labels = np.zeros(n_samples, dtype=int)  # 0 = unvisited
    cluster_id = 0

    def neighbors(i):
        return np.where(D[i] <= eps)[0]

    for i in range(n_samples):
        if labels[i] != 0:  # sudah dikunjungi
            continue
        Ni = neighbors(i)
        if len(Ni) < min_samples:
            labels[i] = -1  # noise sementara
            continue

        # mulai cluster baru
        cluster_id += 1
        labels[i] = cluster_id

        # seed set = tetangga inti (kecuali dirinya)
        seeds = list(Ni[Ni != i])
        idx = 0
        while idx < len(seeds):
            j = seeds[idx]
            if labels[j] == -1:
                labels[j] = cluster_id  # naikkan dari noise jadi border
            if labels[j] == 0:
                labels[j] = cluster_id
                Nj = neighbors(j)
                if len(Nj) >= min_samples:
                    for p in Nj:
                        if p not in seeds:
                            seeds.append(p)
            idx += 1

    return labels


def euclidean_distance_matrix(X):
    """Hitung matriks jarak Euclidean manual antar semua titik."""
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            D[i, j] = dist
            D[j, i] = dist
    return D


def plot_k_distance_graph_manual(X, k=5):
    """Plot k-distance graph tanpa sklearn."""
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
    """Lakukan PCA manual (tanpa sklearn)"""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
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


# === Silhouette added ===
def silhouette_scores_manual(X, labels):
    """
    Hitung silhouette per-sample (manual, Euclidean).
    Mengabaikan label -1 (noise).
    Return:
      s: array silhouette per sample (NaN untuk noise),
      per_cluster: dict {cluster_id: (mean, median, std, n)}
      overall_mean: rata-rata silhouette (tanpa noise & NaN)
    """
    labels = np.asarray(labels)
    n = X.shape[0]
    D = euclidean_distance_matrix(X)

    s = np.full(n, np.nan)
    clusters = [c for c in np.unique(labels) if c != -1]

    if len(clusters) < 1:
        return s, {}, np.nan

    for i in range(n):
        ci = labels[i]
        if ci == -1:
            continue  # noise, biarkan NaN

        same = np.where(labels == ci)[0]
        # a(i): mean intra-cluster distance (exclude self)
        if len(same) <= 1:
            a_i = 0.0
        else:
            a_i = np.mean(D[i, same[same != i]])

        # b(i): minimum mean distance to other clusters
        b_i = np.inf
        for cj in clusters:
            if cj == ci:
                continue
            other = np.where(labels == cj)[0]
            if len(other) == 0:
                continue
            b_i = min(b_i, np.mean(D[i, other]))

        if not np.isfinite(b_i):
            s[i] = 0.0
            continue

        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    # ringkas per-cluster
    per_cluster = {}
    for c in clusters:
        vals = s[labels == c]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            per_cluster[c] = (np.nan, np.nan, np.nan, 0)
        else:
            per_cluster[c] = (float(np.mean(vals)), float(np.median(vals)), float(np.std(vals)), int(len(vals)))

    overall_mean = float(np.nanmean(s))
    return s, per_cluster, overall_mean


# ===================== STREAMLIT APP (TAB UI) =====================

def main():
    st.set_page_config(page_title="Preprocessing & Clustering", page_icon="🧹", layout="wide")
    st.title("🧹 Preprocessing & 🔗 Clustering")

    # ====== TABS LEVEL 1 ======
    tab_prep, tab_km, tab_db = st.tabs(["1) Preprocessing", "2) KMeans", "3) DBSCAN"])

    # ---------------- Preprocessing Tab ----------------
    with tab_prep:
        st.sidebar.header("⚙️ Opsi Preprocessing")
        mode = st.sidebar.selectbox("Mode", ["Improved v2 (disarankan)", "Baseline (awal)"])
        uploaded_file = st.file_uploader("Unggah file CSV/XLSX (Preprocessing)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
            st.subheader("📂 Data Asli (preview)")
            st.dataframe(data.head(124))

            if st.button("🚀 Proses Data"):
                with st.spinner("Memproses..."):
                    if mode == "Improved v2 (disarankan)":
                        imputed, X_scaled, report = manual_preprocess_v2(data)
                    else:
                        result = manual_preprocess_baseline(data)
                        if isinstance(result, tuple) and len(result) == 3:
                            imputed, X_scaled, report = result
                        else:
                            imputed = result
                            X_scaled = imputed.copy()
                            report = {
                                "mode": "baseline",
                                "dropped_cols": [],
                                "onehot_cols": [],
                                "n_rows": int(X_scaled.shape[0]),
                                "n_features": int(X_scaled.shape[1]),
                                "outlier_strategy": None,
                                "scale": None,
                            }

                st.success("✅ Selesai!")
                st.session_state["clean_imputed"] = imputed
                st.session_state["X_scaled"] = X_scaled

                st.subheader("🔎 Data Setelah Preprocessing")
                st.dataframe(imputed.head(124))

                st.subheader("📝 Report")
                st.json(report)

    # ---------------- KMeans Tab ----------------
    with tab_km:
        st.header("🔗 KMeans")

        df_for_cluster = None
        src_km = st.radio("Pilih sumber data KMeans", ["Gunakan hasil Preprocessing", "Upload file baru"], horizontal=True, key="src_km")
        if src_km == "Gunakan hasil Preprocessing":
            if "clean_imputed" in st.session_state:
                df_for_cluster = st.session_state["clean_imputed"].copy()
            else:
                st.warning("Silakan proses data di Tab Preprocessing terlebih dahulu.")
        else:
            uploaded_file2 = st.file_uploader("Unggah file CSV/XLSX untuk KMeans", type=["csv", "xlsx"], key="up_km")
            if uploaded_file2 is not None:
                df_for_cluster = pd.read_csv(uploaded_file2) if uploaded_file2.name.lower().endswith(".csv") else pd.read_excel(uploaded_file2)

        if df_for_cluster is not None:
            st.write("Preview data:")
            st.dataframe(df_for_cluster.head(124))

            # Precompute distance (sesuai logic awal)
            D = gower_distance(df_for_cluster)

            # ====== Sub Tabs KMeans ======
            km_run, km_elbow, km_scatter, km_dist, km_hasil = st.tabs([
                "▶️ Run", "📊 Elbow Method", "🟢 Scatter Plot", "📦 Distribusi", "📌Hasil K-Means Cluster"
            ])

            with km_run:
                k = st.number_input("Jumlah cluster (k)", min_value=2, max_value=30, value=4, key="k_km")
                if st.button("▶️ Jalankan KMeans", key="btn_kmeans_run"):
                    with st.spinner("Clustering..."):
                        kmeans = KMeans(k=k, max_iters=100, random_state=42)
                        kmeans.fit(D)
                        df_for_cluster["cluster"] = kmeans.labels

                        st.success("✅ KMeans selesai!")
                        st.dataframe(df_for_cluster.head(124))

                        st.session_state["km_labels"] = kmeans.labels
                        st.session_state["km_centroids"] = kmeans.centroids
                        st.session_state["km_D"] = D
                        st.session_state["km_df"] = df_for_cluster.copy()

            with km_elbow:
                max_k = st.slider("Maksimum K untuk Elbow Method", min_value=3, max_value=15, value=8, key="maxk_km")
                if st.button("🔍 Jalankan Elbow Method", key="btn_elbow_km"):
                    D_local = st.session_state.get("km_D", gower_distance(df_for_cluster))
                    with st.spinner("Menghitung inertia untuk berbagai K..."):
                        kmeans_eval = KMeans(random_state=42)
                        inertia_values = kmeans_eval.elbow_method(D_local, max_k=max_k)

                    fig_elbow, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(range(1, max_k + 1), inertia_values, marker='o')
                    ax.set_title("Elbow Method untuk Menentukan K Optimal")
                    ax.set_xlabel("Jumlah Cluster (K)")
                    ax.set_ylabel("Inertia (Total Within-Cluster SSE)")
                    ax.grid(True)
                    st.pyplot(fig_elbow)
                    st.info("💡 Titik siku (elbow) pada grafik menunjukkan K optimal.")

            with km_scatter:
                if all(k in st.session_state for k in ["km_labels", "km_centroids", "km_D"]):
                    kmeans_labels = st.session_state["km_labels"]
                    kmeans_centroids = st.session_state["km_centroids"]
                    D_local = st.session_state["km_D"]
                    fig_cluster = KMeans().visualize_clustering_process(
                        D_local, kmeans_labels, kmeans_centroids, df_for_cluster.columns
                    )
                    st.pyplot(fig_cluster)
                else:
                    st.warning("Jalankan KMeans dulu di sub-tab **Run**.")

            with km_dist:
                if "km_df" in st.session_state and "cluster" in st.session_state["km_df"].columns:
                    df_km = st.session_state["km_df"]
                    cluster_counts = df_km["cluster"].value_counts().sort_index()

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_bar, ax1 = plt.subplots(figsize=(4, 3))
                        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax1)
                        ax1.set_xlabel("Cluster")
                        ax1.set_ylabel("Jumlah Data")
                        ax1.set_title("Distribusi Cluster (Bar)")
                        st.pyplot(fig_bar)
                    with col2:
                        fig_pie, ax2 = plt.subplots(figsize=(4, 3))
                        ax2.pie(
                            cluster_counts.values,
                            labels=[f"Cluster {i}" for i in cluster_counts.index],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=sns.color_palette("viridis", len(cluster_counts))
                        )
                        ax2.set_title("Distribusi Cluster (Pie)")
                        st.pyplot(fig_pie)
                else:
                    st.warning("Jalankan KMeans dulu di sub-tab **Run**.")

            with km_hasil:
                st.subheader("📌 Deskripsi Cluster")
                if "km_df" in st.session_state and "cluster" in st.session_state["km_df"].columns:
                    df_km = st.session_state["km_df"]
                    cluster_counts = df_km["cluster"].value_counts().sort_index()

                    descriptions = {
                        0: "Academic-Oriented: Fokus belajar, jarang ikut organisasi.",
                        1: "Balanced: Cukup aktif di akademik & non-akademik.",
                        2: "Non Academic-Oriented: Aktif di UKM/organisasi, tapi belajar minim.",
                        3: "Busy All-Rounder: Aktif di akademik, organisasi, bahkan kerja part-time."
                    }

                    rows = []
                    for cluster_id, count in cluster_counts.items():
                        rows.append({
                            "Cluster": int(cluster_id),
                            "Jumlah": int(count),
                            "Deskripsi": descriptions.get(cluster_id, "Tidak ada deskripsi untuk cluster ini.")
                        })

                    df_desc = pd.DataFrame(rows).sort_values("Cluster")
                    st.dataframe(df_desc, use_container_width=True)

                    st.info(f"Total data: **{len(df_km)}**")
                else:
                    st.warning("Jalankan KMeans dulu di sub-tab **Run**.")

    # ---------------- DBSCAN Tab ----------------
    with tab_db:
        st.header("🔗 DBSCAN")

        df_for_cluster = None
        src_db = st.radio("Pilih sumber data DBSCAN", ["Gunakan hasil Preprocessing", "Upload file baru"], horizontal=True, key="src_db")
        if src_db == "Gunakan hasil Preprocessing":
            if "clean_imputed" in st.session_state:
                df_for_cluster = st.session_state["clean_imputed"].copy()
            else:
                st.warning("Silakan proses data di Tab Preprocessing terlebih dahulu.")
        else:
            uploaded_file3 = st.file_uploader("Unggah file CSV/XLSX untuk DBSCAN", type=["csv", "xlsx"], key="up_db")
            if uploaded_file3 is not None:
                df_for_cluster = pd.read_csv(uploaded_file3) if uploaded_file3.name.lower().endswith(".csv") else pd.read_excel(uploaded_file3)

        if df_for_cluster is not None:
            st.write("Preview data:")
            st.dataframe(df_for_cluster.head(124))

            # Pastikan numerik untuk DBSCAN
            df_numeric = df_for_cluster.select_dtypes(include=["number"]).copy()

            # ====== Sub Tabs DBSCAN ======
            db_kdist, db_run, db_scatter, db_dist, db_hasil = st.tabs([
                "📊 K-Distance Plot", "▶️ Run", "🟢 Scatter (PCA 2D)", "📦 Distribusi", "📌 Hasil DBSCAN"
            ])

            # util: pilih matriks data yang dipakai (X_scaled atau numerik mentah) + opsi PCA
            def _prepare_matrix(use_scaled_key:str, use_pca_key:str, ncomp_key:str):
                M = df_numeric.values
                use_scaled = st.checkbox("Gunakan X_scaled dari preprocessing (disarankan)", 
                                         value=("X_scaled" in st.session_state), key=use_scaled_key)
                if use_scaled and ("X_scaled" in st.session_state):
                    M = st.session_state["X_scaled"].values

                use_pca = st.checkbox("PCA sebelum DBSCAN", value=False, key=use_pca_key)
                if use_pca:
                    max_comp = max(2, min(M.shape[1], 50))
                    ncomp = st.slider("Jumlah komponen PCA", min_value=2, max_value=max_comp,
                                      value=min(10, max_comp), key=ncomp_key)
                    M = pca_manual(M, n_components=ncomp)
                    st.caption(f"Matrix untuk proses (setelah opsi): shape = {M.shape}")
                else:
                    st.caption(f"Matrix untuk proses: shape = {M.shape}")
                return M

            # ---------- K-Distance Plot ----------
            with db_kdist:
                if df_numeric.empty:
                    st.error("❌ Tidak ada kolom numerik yang bisa digunakan untuk DBSCAN!")
                else:
                    M_k = _prepare_matrix("db_use_scaled_k", "db_use_pca_k", "db_ncomp_k")
                    k_value = st.slider("Pilih nilai k (biasanya = min_samples)", min_value=3, max_value=50, value=6, key="k_db")
                    if st.button("🔍 Tampilkan K-Distance Plot", key="btn_kdist"):
                        with st.spinner("Menghitung k-distance..."):
                            fig, k_dist = plot_k_distance_graph_manual(M_k, k=k_value)
                            st.pyplot(fig)
                            st.info("💡 Cari 'titik siku' (elbow). Nilai Y di titik siku ≈ **eps optimal**.\n"
                                    "   Tips: samakan k dengan `min_samples` yang akan kamu pakai di tab Run.")

            # ---------- Run DBSCAN ----------
            with db_run:
                if df_numeric.empty:
                    st.error("❌ Tidak ada kolom numerik yang bisa digunakan untuk DBSCAN!")
                else:
                    M_run = _prepare_matrix("db_use_scaled_run", "db_use_pca_run", "db_ncomp_run")

                    colp1, colp2 = st.columns(2)
                    with colp1:
                        eps = st.number_input("eps", min_value=0.01, max_value=5.0, value=0.35, step=0.05, key="eps_db",
                                              help="Radius maksimum untuk mencari tetangga (gunakan acuan K-Distance Plot).")
                    with colp2:
                        min_samples = st.number_input("min_samples", min_value=3, max_value=100, value=6, step=1, key="mins_db",
                                                      help="Jumlah minimum tetangga untuk membentuk cluster (core).")

                    if st.button("▶️ Jalankan DBSCAN", key="btn_dbscan_run"):
                        with st.spinner("Clustering..."):
                            db_labels = dbscan(M_run, eps=eps, min_samples=min_samples)

                            unique_labels = np.unique(db_labels)
                            n_clusters = int(np.sum(unique_labels != -1))
                            n_noise = int(np.sum(db_labels == -1))

                            # simpan hasil
                            df_for_cluster["cluster"] = db_labels
                            st.session_state["db_labels"] = db_labels
                            st.session_state["db_df"] = df_for_cluster.copy()
                            st.session_state["db_matrix_used"] = M_run
                            st.session_state["db_params"] = {
                                "eps": float(eps),
                                "min_samples": int(min_samples),
                                "n_clusters": n_clusters,
                                "n_noise": n_noise,
                                "shape": tuple(M_run.shape)
                            }

                            st.success(f"✅ DBSCAN selesai! **{n_clusters} cluster** ditemukan, **{n_noise} noise points**")
                            st.write("**Distribusi Cluster (counts):**")
                            cluster_counts = pd.Series(db_labels).value_counts().sort_index()
                            st.dataframe(cluster_counts.to_frame("Jumlah"))
                            st.dataframe(df_for_cluster.head(124))

            # ---------- Scatter (PCA 2D hanya untuk visualisasi) ----------
            with db_scatter:
                if all(k in st.session_state for k in ["db_labels", "db_matrix_used"]):
                    labels = st.session_state["db_labels"]
                    M_used = st.session_state["db_matrix_used"]

                    fig_dbscan = visualize_dbscan_results(M_used, labels)
                    st.pyplot(fig_dbscan)

                    if "db_params" in st.session_state:
                        p = st.session_state["db_params"]
                        st.caption(f"Param: eps={p['eps']}, min_samples={p['min_samples']} • "
                                   f"Clusters={p['n_clusters']} • Noise={p['n_noise']} • Matrix={p['shape']}")
                else:
                    st.warning("Jalankan DBSCAN dulu di sub-tab **Run**.")

            # ---------- Distribusi (bar & pie) ----------
            with db_dist:
                if "db_df" in st.session_state and "cluster" in st.session_state["db_df"].columns:
                    df_db = st.session_state["db_df"]
                    cluster_counts = df_db["cluster"].value_counts().sort_index()

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_bar, ax1 = plt.subplots(figsize=(4, 3))
                        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax1)
                        ax1.set_xlabel("Cluster")
                        ax1.set_ylabel("Jumlah Data")
                        ax1.set_title("Distribusi Cluster (Bar)")
                        st.pyplot(fig_bar)
                    with col2:
                        fig_pie, ax2 = plt.subplots(figsize=(4, 3))
                        ax2.pie(
                            cluster_counts.values,
                            labels=[f"Cluster {i}" for i in cluster_counts.index],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=sns.color_palette("viridis", len(cluster_counts))
                        )
                        ax2.set_title("Distribusi Cluster (Pie)")
                        st.pyplot(fig_pie)
                else:
                    st.warning("Jalankan DBSCAN dulu di sub-tab **Run**.")

            # ---------- Hasil Ringkas + Silhouette ----------
            with db_hasil:
                st.subheader("📌 Deskripsi Hasil DBSCAN")
                if "db_df" in st.session_state and "cluster" in st.session_state["db_df"].columns:
                    df_db = st.session_state["db_df"].copy()
                    counts = df_db["cluster"].value_counts().sort_index()
                    total = int(len(df_db))
                    summary_rows = []
                    for cid, cnt in counts.items():
                        pct = (cnt / total) * 100 if total > 0 else 0.0
                        summary_rows.append({
                            "Cluster": int(cid),
                            "Jumlah": int(cnt),
                            "Persentase (%)": round(pct, 2),
                            "Keterangan": "Noise" if cid == -1 else f"Cluster {int(cid)}"
                        })
                    df_summary = pd.DataFrame(summary_rows).sort_values(["Cluster"])
                    st.dataframe(df_summary, use_container_width=True)

                    # === Silhouette tampil di sini ===
                    if all(k in st.session_state for k in ["db_labels", "db_matrix_used"]):
                        labels = st.session_state["db_labels"]
                        M_used = st.session_state["db_matrix_used"]

                        # Hitung silhouette (manual)
                        s_values, per_cluster, overall = silhouette_scores_manual(M_used, labels)

                        # Tabel ringkas per cluster (exclude noise)
                        rows_sil = []
                        for c, (mean_c, med_c, std_c, n_c) in sorted(per_cluster.items()):
                            rows_sil.append({
                                "Cluster": int(c),
                                "Silhouette Mean": round(mean_c, 4) if not np.isnan(mean_c) else np.nan,
                                "Silhouette Median": round(med_c, 4) if not np.isnan(med_c) else np.nan,
                                "Silhouette Std": round(std_c, 4) if not np.isnan(std_c) else np.nan,
                                "N": n_c
                            })
                        df_sil = pd.DataFrame(rows_sil)
                        st.subheader("📈 Silhouette Score (Manual)")
                        st.write(f"**Overall silhouette (tanpa noise)**: **{overall:.4f}**" if np.isfinite(overall) else "Overall silhouette tidak terdefinisi.")
                        if not df_sil.empty:
                            st.dataframe(df_sil, use_container_width=True)

                            # Bar chart mean silhouette per cluster
                            fig_sil, ax_sil = plt.subplots(figsize=(5, 3))
                            sns.barplot(data=df_sil, x="Cluster", y="Silhouette Mean", ax=ax_sil, palette="viridis")
                            ax_sil.set_ylim(-1, 1)
                            ax_sil.set_title("Mean Silhouette per Cluster (exclude noise)")
                            ax_sil.grid(True, axis='y', alpha=0.3)
                            st.pyplot(fig_sil)
                        else:
                            st.info("Tidak ada cluster valid untuk dihitung silhouette (mungkin semua noise atau hanya 1 cluster).")
                    # === end Silhouette ===

                    # === Penjelasan "Noise" (tabel + poin) ===
                if "db_df" in st.session_state and "cluster" in st.session_state["db_df"].columns:
                    df_db_local = st.session_state["db_df"].copy()
                    total_n = int(len(df_db_local))
                    n_noise = int((df_db_local["cluster"] == -1).sum())
                    pct_noise = (n_noise / total_n * 100) if total_n > 0 else 0.0

                    st.subheader("🧩 Ringkasan Noise (Outlier)")

                    # Tabel ringkas noise
                    noise_table = pd.DataFrame([
                        {"Aspek": "Total Data", "Keterangan": total_n},
                        {"Aspek": "Jumlah Noise (-1)", "Keterangan": n_noise},
                        {"Aspek": "Persentase Noise", "Keterangan": f"{pct_noise:.2f}%"},
                        {"Aspek": "Definisi Singkat", "Keterangan": "Data yang tidak cukup mirip dengan kelompok manapun sehingga dianggap pencilan (outlier)."},
                    ])
                    st.table(noise_table)

                    # Poin penjelasan & rekomendasi
                    st.markdown("""
                **Kenapa bisa muncul noise?**
                - Pola/karakteristik **sangat berbeda** dari mayoritas (mis. waktu belajar sangat ekstrem).
                - **Jawaban survei tidak lengkap** atau tidak wajar (nilai di luar rentang, semua jawaban sama).
                - **Sangat unik secara positif** (super aktif/berprestasi) sehingga jauh dari pola umum.
                - **Kualitas data**: ada kesalahan input atau inkonsistensi.

                **Apa yang sebaiknya dilakukan?**
                - **Validasi data**: cek apakah ini kasus unik nyata atau ada kesalahan pengisian.
                - Jika unik **positif**, pertimbangkan jadi **role model/mentor** untuk berbagi strategi belajar.
                - Jika mengindikasikan **tantangan akademik/psikologis**, lakukan **pendampingan** (bimbingan/konseling).
                - **Perbaiki instrumen**: tinjau pertanyaan survei, rentang jawaban, dan cara pengumpulan agar lebih konsisten.
                """)
                # === end penjelasan noise ===

    
                    if "db_params" in st.session_state:
                        p = st.session_state["db_params"]
                        st.info(
                            f"Parameter: eps = **{p['eps']}**, min_samples = **{p['min_samples']}** • "
                            f"Clusters = **{p['n_clusters']}**, Noise = **{p['n_noise']}** • "
                            f"Matrix untuk proses = **{p['shape']}** • Total data = **{total}**"
                        )
                else:
                    st.warning("Jalankan DBSCAN dulu di sub-tab **Run**.")

if __name__ == "__main__":
    main()
