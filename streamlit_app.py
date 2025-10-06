import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def initialize_centroids_kmeans_pp(X, k, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    centroids = []

    centroids.append(X[np.random.randint(0, n_samples)])

    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
        prob = distances / distances.sum()
        cumulative_prob = np.cumsum(prob)
        r = np.random.rand()
        next_centroid_idx = np.searchsorted(cumulative_prob, r)
        centroids.append(X[next_centroid_idx])

    return np.array(centroids)

def kmeans_manual_pp(X, k=4, max_iters=300, tol=1e-5, random_state=42):
    centroids = initialize_centroids_kmeans_pp(X, k, random_state)

    for _ in range(max_iters):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids

    return labels, centroids

st.set_page_config(page_title="Survei Keseimbangan Aktivitas Mahasiswa", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #1f2937; font-weight: 800;'>📊 Survei Keseimbangan Aktivitas Mahasiswa</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #6b7280; font-size:18px;'>Analisis clustering untuk memahami keseimbangan kegiatan akademik dan non akademik mahasiswa</p>",
    unsafe_allow_html=True
)
st.markdown("---")

uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("📂 Data yang Diunggah")
    st.dataframe(data, use_container_width=True)

    mapping = {
        'Kurang dari 1 jam': 0.5,
        '1 - 2 jam': 1.5,
        '3 - 5 jam': 4,
        '6 - 8 jam': 7,
        'Lebih dari 12 jam': 12,
        'Selalu': 4,
        'Sering': 3,
        'Kadang-kadang': 2,
        'Jarang': 1,
        'Tidak pernah': 0,
        'Sangat penting': 4,
        'Penting': 3,
        'Cukup penting': 2,
        'Tidak terlalu penting': 1,
        'Ya': 1,
        'Tidak': 0,
        'Sangat seimbang': 4,
        'Seimbang': 3,
        'Kurang seimbang': 2,
        'Tidak seimbang': 1
    }

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].map(mapping)

def manual_preprocess(data):
    # 1️⃣ Mapping teks ke angka
    mapping = {
        'Kurang dari 1 jam': 0.5, '1 - 2 jam': 1.5, '3 - 5 jam': 4, '6 - 8 jam': 7, 'Lebih dari 12 jam': 12,
        'Selalu': 4, 'Sering': 3, 'Kadang-kadang': 2, 'Jarang': 1, 'Tidak pernah': 0,
        'Sangat penting': 4, 'Penting': 3, 'Cukup penting': 2, 'Tidak terlalu penting': 1,
        'Ya': 1, 'Tidak': 0,
        'Sangat seimbang': 4, 'Seimbang': 3, 'Kurang seimbang': 2, 'Tidak seimbang': 1
    }

    df = data.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(mapping)

    # 2️⃣ Ambil kolom numerik
    df_num = df.select_dtypes(include=['number']).copy()

    # 3️⃣ Imputasi manual (pakai median)
    for col in df_num.columns:
        median_val = df_num[col].median()
        df_num[col] = df_num[col].fillna(median_val)

    # 4️⃣ Deteksi & buang outlier (IQR manual)
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before_rows = len(df_num)
    df_filtered = df_num[~((df_num < lower_bound) | (df_num > upper_bound)).any(axis=1)]
    outlier_removed = before_rows - len(df_filtered)

    # 5️⃣ Log-transform bila distribusi terlalu miring
    # (cek skewness manual)
    for col in df_filtered.columns:
        skewness = ((df_filtered[col] - df_filtered[col].mean())**3).mean() / (df_filtered[col].std()**3)
        if abs(skewness) > 1:  # data terlalu miring
            df_filtered[col] = np.log1p(df_filtered[col] - df_filtered[col].min() + 1)

    # 6️⃣ Normalisasi manual (z-score)
    mean_vals = df_filtered.mean()
    std_vals = df_filtered.std(ddof=0)
    X_scaled = (df_filtered - mean_vals) / std_vals

    return df_filtered, X_scaled, outlier_removed


# 🧩 Ganti bagian lama preprocessing kamu:
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("📂 Data yang Diunggah")
    st.dataframe(data, use_container_width=True)

    # 🚀 Preprocessing manual
    data_imputed, X_scaled, outlier_removed = manual_preprocess(data)
    st.success(f"✅ Data berhasil diproses — {outlier_removed} baris dihapus karena outlier.")

    # Tampilkan preview hasil preprocessing
    st.subheader("🔎 Data Setelah Preprocessing")
    st.dataframe(data_imputed.round(3), use_container_width=True)

    if st.button("🚀 Proses Data dan Kelompokkan (Manual KMeans++)"):
        labels, centroids = kmeans_manual_pp(X_scaled, k=4)
        data_imputed['Cluster'] = labels

        data_imputed_rounded = data_imputed.round(2)
        st.subheader("📊 Hasil Clustering")

        def highlight_row(val):
            colors = {
                0: 'background-color: #93c5fd; color: black;',   
                1: 'background-color: #86efac; color: black;',   
                2: 'background-color: #d8b4fe; color: black;',   
                3: 'background-color: #fdba74; color: black;'    
            }
            return colors.get(val, '')

        styled_df = data_imputed.style.applymap(highlight_row, subset=['Cluster'])
        st.dataframe(styled_df, use_container_width=True, height=300)

        cluster_summary = {
            0: "Cluster 1 – Academic-Oriented: Fokus belajar, jarang ikut organisasi.",
            1: "Cluster 2 – Balanced: Cukup aktif di akademik & non-akademik.",
            2: "Cluster 3 – Non Academic-Oriented: Aktif di UKM, organisasi, tapi belajar minim.",
            3: "Cluster 4 – Busy All-Rounder: Aktif di akademik, organisasi, bahkan kerja part-time."
        }
        cluster_counts = data_imputed['Cluster'].value_counts()
        st.subheader("📌 Deskripsi Cluster")
        for i in range(4):
            count = cluster_counts.get(i, 0)
            st.write(f"**{cluster_summary[i]}** (Jumlah: {count} mahasiswa)")

        st.subheader("📊 Visualisasi Cluster (Plotly)")
        if 'Jam Belajar' in data_imputed.columns and 'Jam Organisasi' in data_imputed.columns:
            fig = px.scatter(
                data_frame=data_imputed,
                x='Jam Belajar',
                y='Jam Organisasi',
                color='Cluster',
                labels={'Jam Belajar': 'Jam Belajar per Minggu', 'Jam Organisasi': 'Jam Organisasi per Minggu'},
                title="Visualisasi Cluster (Manual KMeans++)"
            )
            st.plotly_chart(fig)

        st.subheader("🎯 Visualisasi Cluster (Seaborn)")
        if 'Jam Belajar' in data_imputed.columns and 'Jam Organisasi' in data_imputed.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                x=data_imputed['Jam Belajar'],
                y=data_imputed['Jam Organisasi'],
                hue=data_imputed['Cluster'],
                palette="Set1",
                s=120,
                alpha=0.8,
                ax=ax
            )
            for i, centroid in enumerate(centroids):
                try:
                    ax.scatter(
                        centroid[data_imputed.columns.get_loc('Jam Belajar')],
                        centroid[data_imputed.columns.get_loc('Jam Organisasi')],
                        marker='X',
                        s=200,
                        color='black',
                        edgecolor='white',
                        label=f'Centroid {i}'
                    )
                except:
                    pass
            plt.title("Cluster Berdasarkan Jam Belajar & Jam Organisasi")
            ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
            st.pyplot(fig)

        st.subheader("🌀 Visualisasi PCA 2D")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        data_imputed["Fokus Akademik"] = X_pca[:, 0]
        data_imputed["Kegiatan Sosial"] = X_pca[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=data_imputed,
            x="Fokus Akademik",
            y="Kegiatan Sosial",
            hue="Cluster",
            palette="Set2",
            s=120,
            alpha=0.8
        )
        for i, centroid in enumerate(centroids):
            centroid_pca = pca.transform(centroid.reshape(1, -1))
            ax.scatter(
                centroid_pca[0, 0],
                centroid_pca[0, 1],
                marker='X',
                s=200,
                color='black',
                edgecolor='white'
            )
        plt.title("Visualisasi Cluster Mahasiswa (PCA 2D)")
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)
