# streamlit_app.py
# ===========================================================================================
# PREPROCESSING + CLUSTERING (manual)
# - Tab 1: Preprocessing (Baseline & Improved v2)
# - Tab 2: Clustering manual (Agglomerative Average-Linkage & DBSCAN) dgn Gower manual
# Jalankan: streamlit run streamlit_app.py
# Dependensi: pip install streamlit pandas numpy matplotlib openpyxl
# ===========================================================================================

import pandas as pd
import numpy as np
import streamlit as st

# Import semua util dari data_preprocessing.py
from data_preprocessing import (
    # Preprocess
    manual_preprocess_baseline, manual_preprocess_v2,
    # Distance & metrics
    gower_distance, silhouette_precomputed,
    # Clustering
    agglomerative_average, dbscan_precomputed,
    # Plots
    k_distance_values, plot_k_distance_curve, plot_merge_distances,
    plot_clusters_2d,   
)

st.set_page_config(page_title="Preprocessing & Clustering (Manual)", page_icon="🧹", layout="wide")

# ---------------- Helpers kecil untuk UI ----------------
def to_csv_download_button(df: pd.DataFrame, filename: str, label: str):
    """Tombol download CSV di Streamlit."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ==================== UI ====================
st.title("🧹 Preprocessing & 🔗 Clustering (Manual)")

tab1, tab2 = st.tabs(["1) Preprocessing", "2) Clustering (Manual)"])

# ---------------- TAB 1: PREPROCESSING ----------------
with tab1:
    st.sidebar.header("⚙️ Opsi Preprocessing")
    mode = st.sidebar.selectbox("Mode", ["Improved v2 (disarankan)", "Baseline (awal)"])
    if mode == "Improved v2 (disarankan)":
        outlier_mode = st.sidebar.selectbox("Outlier handling", ["clip", "drop"])
        scale_mode = st.sidebar.selectbox("Scaling", ["robust", "standard", "None"])
        small_cat_max = st.sidebar.number_input("Max kardinalitas kategori kecil (one-hot)", 2, 50, 12, 1)
        skew_th = st.sidebar.slider("Ambang log-transform (|skew|>", 0.5, 3.0, 1.0, 0.1)
    else:
        outlier_mode = "drop"; scale_mode = "standard"; small_cat_max = 12; skew_th = 1.0

    uploaded_file = st.file_uploader("Unggah file CSV/XLSX (Preprocessing)", type=["csv", "xlsx"], key="uploader_pre")

    if uploaded_file is not None:
        # Baca file
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)  # butuh openpyxl
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.subheader("📂 Data Asli (preview)")
        st.dataframe(data.head(15), use_container_width=True)

        if st.button("🚀 Proses Data", key="process_pre"):
            with st.spinner("Memproses..."):
                try:
                    if mode == "Improved v2 (disarankan)":
                        imputed, X_scaled, report = manual_preprocess_v2(
                            data,
                            outlier=outlier_mode,
                            scale=None if scale_mode == "None" else scale_mode,
                            small_cat_max_card=small_cat_max,
                            skew_thresh=skew_th
                        )
                    else:
                        imputed, X_scaled, report = manual_preprocess_baseline(data)
                except Exception as e:
                    st.exception(e)
                    st.stop()

            st.success("✅ Selesai!")
            # simpan juga X_scaled ke session untuk opsi plot
            st.session_state["X_scaled"] = X_scaled.to_numpy()

            # Simpan hasil clean/imputed ke session supaya bisa dipakai di Tab Clustering
            st.session_state["clean_imputed"] = imputed

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔎 Data Setelah Preprocessing (imputed/clean)")
                st.dataframe(imputed.head(25), use_container_width=True)
                to_csv_download_button(imputed, "clean_imputed.csv", "⬇️ Download clean_imputed.csv")
            with c2:
                st.subheader("📐 Fitur Terskalakan (X_scaled)")
                st.dataframe(X_scaled.head(25), use_container_width=True)
                to_csv_download_button(X_scaled, "X_scaled.csv", "⬇️ Download X_scaled.csv")

            st.subheader("📝 Report")
            st.json(report)
    else:
        st.info("Silakan unggah file terlebih dahulu untuk preprocessing.")

# ---------------- TAB 2: CLUSTERING MANUAL ----------------
with tab2:
    st.header("🔗 Clustering (Agglomerative / DBSCAN — Manual)")

    # Pilih sumber data: pakai hasil preprocessing atau upload baru
    src = st.radio("Pilih sumber data clustering", ["Gunakan hasil Preprocessing (clean_imputed)", "Upload file baru"], horizontal=True)
    df_for_cluster = None

    if src == "Gunakan hasil Preprocessing (clean_imputed)":
        if "clean_imputed" in st.session_state:
            df_for_cluster = st.session_state["clean_imputed"].copy()
            st.success(f"Memakai data dari preprocessing: shape = {df_for_cluster.shape}")
        else:
            st.warning("Belum ada hasil preprocessing. Silakan proses di Tab 1 atau upload file baru.")
    else:
        up2 = st.file_uploader("Unggah file CSV/XLSX (untuk Clustering)", type=["csv", "xlsx"], key="uploader_cluster")
        if up2 is not None:
            try:
                if up2.name.lower().endswith(".csv"):
                    df_for_cluster = pd.read_csv(up2)
                else:
                    df_for_cluster = pd.read_excel(up2)
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    if df_for_cluster is not None:
        st.write("Preview data untuk clustering:")
        st.dataframe(df_for_cluster.head(10), use_container_width=True)

        method = st.selectbox("Metode", ["Agglomerative (Average-Linkage, Manual)", "DBSCAN (Manual)"])

        # Hitung matriks jarak Gower manual (pakai data bersih/imputed)
        with st.spinner("Menghitung matriks jarak (Gower manual)..."):
            D = gower_distance(df_for_cluster)

        # -------- Agglomerative manual --------
        if method == "Agglomerative (Average-Linkage, Manual)":
            k = st.number_input("Jumlah cluster (k)", min_value=2, max_value=30, value=4, step=1)
            if st.button("▶️ Jalankan Agglomerative (Manual)"):
                with st.spinner("Clustering..."):
                    labels, merges = agglomerative_average(D, int(k))
                    sil = silhouette_precomputed(D, labels)

                st.success(f"Selesai. Silhouette (precomputed) = {sil:.3f}")
                df_out = df_for_cluster.copy()
                df_out["cluster"] = labels
                st.dataframe(df_out.head(20), use_container_width=True)
                to_csv_download_button(df_out, f"clusters_agg_manual_k{k}.csv", f"⬇️ Download clusters_agg_manual_k{k}.csv")

                st.subheader("📈 Kurva jarak penggabungan (indikasi lompatan untuk pilih k)")
                fig = plot_merge_distances(merges)
                if fig is not None:
                    st.pyplot(fig)

                st.subheader("📊 Ringkasan per cluster (median tiap fitur)")
                st.dataframe(df_out.groupby("cluster").median(numeric_only=True), use_container_width=True)

        # -------- DBSCAN manual --------
        else:
            c1, c2 = st.columns(2)
            with c1:
                eps = st.number_input("eps (0.01–1.0)", min_value=0.01, max_value=5.0, value=0.35, step=0.01)
            with c2:
                min_samples = st.number_input("min_samples", min_value=3, max_value=50, value=6, step=1)

            if st.button("▶️ Jalankan DBSCAN (Manual)"):
                with st.spinner("Clustering..."):
                    labels = dbscan_precomputed(D, float(eps), int(min_samples))
                    valid = labels != -1
                    if np.unique(labels[valid]).size >= 2:
                        sil = silhouette_precomputed(D[np.ix_(valid, valid)], labels[valid])
                    else:
                        sil = float("nan")
                    noise_pct = (labels == -1).mean() * 100

                st.success(f"Selesai. Silhouette(valid) = {sil if sil==sil else 'NaN'} | Noise = {noise_pct:.1f}%")
                df_out = df_for_cluster.copy()
                df_out["cluster"] = labels
                st.dataframe(df_out.head(20), use_container_width=True)
                to_csv_download_button(df_out, f"clusters_dbscan_manual.csv", "⬇️ Download clusters_dbscan_manual.csv")

            st.subheader("📈 k-distance plot (bantu pilih eps)")
            k_nn = st.number_input("k untuk k-distance (≈ min_samples)", min_value=3, max_value=50, value=int(min_samples), step=1)
            if st.button("Tampilkan k-distance plot"):
                vals = k_distance_values(D, int(k_nn))
                fig = plot_k_distance_curve(vals, int(k_nn))
                st.pyplot(fig)
    else:
        st.info("Pilih sumber data dan/atau upload file untuk melakukan clustering.")
