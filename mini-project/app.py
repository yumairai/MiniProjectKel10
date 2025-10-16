import pandas as pd
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import manual_preprocess_baseline, manual_preprocess_v2
from clustering_kmeans import KMeans, gower_distance, visualize_clustering_process, elbow_method
from clustering_dbscan import (
    dbscan, plot_k_distance_graph_manual, visualize_dbscan_results, pca_manual
)
from metrics import silhouette_scores_manual
from components import render_noise_section, bar_pie_distribution

def main():
    st.set_page_config(
        page_title="Campus Activity Insight (CAI)",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Campus Activity Insight (CAI)")
    st.markdown("🔍 Analisis Keseimbangan Aktivitas Mahasiswa — Preprocessing & Clustering")

    tab_prep, tab_km, tab_db = st.tabs([
        "Preprocessing",
        "K-Means Clustering",
        "DBSCAN Clustering"
    ])


    # ---------------- Preprocessing Tab ----------------
    with tab_prep:
        # ===================== 🌟 Sidebar Design =====================
        st.sidebar.markdown("## 🎓 **Campus Activity Insight (CAI)**")
        st.sidebar.markdown(
            """
            📊 **CAI** membantu menganalisis *keseimbangan aktivitas mahasiswa*  
            antara **akademik** dan **non-akademik**,  
            menggunakan metode **K-Means** dan **DBSCAN**.

            ---
            🔹 **Langkah-langkah:**
            1. Unggah dataset atau gunakan dataset default  
            2. Lakukan *Preprocessing*  
            3. Jalankan *Clustering (K-Means / DBSCAN)*  
            4. Lihat hasil dan interpretasi tiap cluster

            ---
            💡 *Gunakan hasil preprocessing untuk clustering agar hasil lebih akurat.*
            """
        )

        st.sidebar.markdown("---")
        st.header("📂 Upload atau Pilih Dataset")

        data_source = st.radio(
        "Pilih sumber data:",
        ["Upload Dataset", "Gunakan Dataset Default"],
        index=0,
        horizontal=True
        )

        data = None

        # === Opsi 1: Upload Sendiri ===
        if data_source == "Upload Dataset":
            uploaded_file = st.file_uploader("Unggah file CSV/XLSX (Preprocessing)", type=["csv", "xlsx"])
            if uploaded_file is not None:
                if uploaded_file.name.lower().endswith(".csv"):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)

        # === Opsi 2: Gunakan Dataset Default ===
        else:
            try:
                base_dir = os.path.dirname(__file__)  # lokasi file app.py
                default_path = os.path.join(
                    base_dir,
                    "Survei Keseimbangan Aktivitas Mahasiswa (Responses).xlsx"
                )

                data = pd.read_excel(default_path)
                st.success("✅ Menggunakan dataset default bawaan.")
            except Exception as e:
                st.error(f"❌ Gagal memuat dataset default: {e}")
                st.info("Silakan unggah file atau pastikan file default tersedia di folder 'mini-project/'.")

        # === Jika data tersedia, tampilkan preview ===
        if data is not None:
            st.subheader("📂 Data (preview)")
            st.dataframe(data.head(124))

            # === Tombol Proses Data ===
            if st.button("🚀 Proses Data"):
                with st.spinner("Memproses..."):
                    imputed, X_scaled, report = manual_preprocess_v2(data)

                st.success("✅ Selesai!")
                st.session_state["clean_imputed"] = imputed
                st.session_state["X_scaled"] = X_scaled

                st.subheader("🔎 Data Setelah Preprocessing")
                st.dataframe(imputed.head(124))

                st.subheader("📝 Report")
                st.json(report)

        else:
            st.info("📁 Silakan unggah file atau gunakan dataset default untuk melanjutkan.")
    # ---------------- KMeans Tab ----------------
    with tab_km:
        st.header("🔗 KMeans")
        df_for_cluster = None
        src_km = st.radio("Pilih sumber data KMeans", ["Gunakan hasil Preprocessing", "Upload file baru"], horizontal=True, key="src_km")
        if src_km == "Gunakan hasil Preprocessing":
            if "X_scaled" in st.session_state:
                df_for_cluster = st.session_state["X_scaled"].copy()
            elif "clean_imputed" in st.session_state:
                st.warning("⚠️ Data belum diskalakan, menggunakan clean_imputed sebagai cadangan.")
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
            df_for_analysis = df_for_cluster.copy()
            cols_to_drop = [c for c in df_for_cluster.columns if "mean" in c.lower()]
            if cols_to_drop:
                df_for_cluster = df_for_cluster.drop(columns=cols_to_drop)

            D = gower_distance(df_for_cluster)

            km_run, km_elbow_tab, km_scatter, km_dist, km_hasil = st.tabs([
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

                        try:
                            s_values_km, per_cluster_km, overall_km = silhouette_scores_manual(D, kmeans.labels)
                        except Exception as e:
                            s_values_km, per_cluster_km, overall_km = None, {}, np.nan
                            st.warning(f"Gagal menghitung Silhouette untuk K-Means: {e}")

                                # Menyimpan hasil silhouette score ke session_state
                        st.session_state["km_sil"] = {
                            "s_values": s_values_km,
                            "per_cluster": per_cluster_km,
                            "overall": float(overall_km) if overall_km is not None and np.isfinite(overall_km) else None,
                            }
                        
            with km_elbow_tab:
                max_k = st.slider("Maksimum K untuk Elbow Method", min_value=3, max_value=15, value=8, key="maxk_km")
                if st.button("🔍 Jalankan Elbow Method", key="btn_elbow_km"):
                    D_local = st.session_state.get("km_D", gower_distance(df_for_cluster))
                    with st.spinner("Menghitung inertia untuk berbagai K..."):
                        inertia_values = elbow_method(D_local, max_k=max_k, random_state=42)

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
                    fig_cluster = visualize_clustering_process(
                        st.session_state["km_D"],
                        st.session_state["km_labels"],
                        st.session_state["km_centroids"]
                    )
                    st.pyplot(fig_cluster)
                else:
                    st.warning("Jalankan KMeans dulu di sub-tab **Run**.")

            with km_dist:
                if "km_df" in st.session_state and "cluster" in st.session_state["km_df"].columns:
                    cluster_counts = st.session_state["km_df"]["cluster"].value_counts().sort_index()
                    bar_pie_distribution(cluster_counts)
                else:
                    st.warning("Jalankan KMeans dulu di sub-tab **Run**.")

            with km_hasil:
                st.subheader("📌 Deskripsi Cluster")
                if "km_df" in st.session_state and "cluster" in st.session_state["km_df"].columns:
                    df_km = st.session_state["km_df"]
                    cluster_counts = df_km["cluster"].value_counts().sort_index()
                    hasil_results = {}

                    km_sil = st.session_state.get("km_sil", {})
                    overall_silhouette = km_sil.get("overall", None)

                    if overall_silhouette is not None:
                        st.markdown(f"**Overall Silhouette Score**: **{overall_silhouette:.4f}**")
                    else:
                        st.info("Silhouette score tidak tersedia. Pastikan K-Means sudah dijalankan.")


                    if "per_cluster" in km_sil:
                        per_cluster = km_sil["per_cluster"]
                        silhouette_means = [mean_c for mean_c, _, _, _ in per_cluster.values()]
                        cluster_ids = list(per_cluster.keys())
                        
                        fig_silhouette, ax_silhouette = plt.subplots(figsize=(5, 3))
                        ax_silhouette.bar(cluster_ids, silhouette_means, color='skyblue')
                        ax_silhouette.set_xlabel("Cluster")
                        ax_silhouette.set_ylabel("Silhouette Mean")
                        ax_silhouette.set_title("Silhouette Score per Cluster")
                        ax_silhouette.set_ylim(-1, 1)  # Nilai Silhouette berada dalam rentang -1 hingga 1
                        ax_silhouette.grid(True, axis='y', linestyle='--', alpha=0.7)
                        st.pyplot(fig_silhouette)

                    # --- Loop tiap cluster
                    for cluster_id in sorted(cluster_counts.index):
                        df_cluster = df_km[df_km["cluster"] == cluster_id]

                        akademik_cols = [
                            "Berapa jam rata-rata per minggu kamu gunakan untuk belajar mandiri (di luar kelas)?",
                            "Seberapa sering kamu mengerjakan tugas tepat waktu?",
                            "Seberapa sering kamu mengikuti kegiatan akademik tambahan (kuliah tamu, seminar, workshop)?",
                            "Bagaimana tingkat prioritasmu terhadap IPK?"
                        ]

                        nonak_cols = [
                            "Apakah kamu aktif mengikuti organisasi/UKM di kampus?",
                            "Berapa jam rata-rata per minggu untuk organisasi/UKM?",
                            "Apakah kamu bekerja part-time/freelance?",
                            "Berapa jam rata-rata per minggu untuk pekerjaan/hobi/olahraga?",
                            "Seberapa penting kegiatan non-akademik bagimu?"
                        ]

                        # Hitung rata-rata dua kolom per cluster
                        mean_ipk = df_cluster["Bagaimana tingkat prioritasmu terhadap IPK?"].mean()
                        mean_nonak = df_cluster["Seberapa penting kegiatan non-akademik bagimu?"].mean()

                        # Bandingkan untuk tentukan prioritas dominan
                        if mean_ipk > mean_nonak:
                            prioritas = "Akademik"
                        else:
                            prioritas = "Non-Akademik"

                        # --- Ambil kolom akademik & non-akademik yang ada ---
                        akademik_exist = [c for c in akademik_cols if c in df_cluster.columns]
                        nonak_exist = [c for c in nonak_cols if c in df_cluster.columns]

                        # --- Hitung mean mentah per baris ---
                        mean_akademik_raw = df_cluster[akademik_exist].mean(axis=1)
                        mean_nonak_raw = df_cluster[nonak_exist].mean(axis=1)

                        # --- Normalisasi manual ke skala 0–1 ---
                        mean_akademik = (mean_akademik_raw - mean_akademik_raw.min()) / (mean_akademik_raw.max() - mean_akademik_raw.min())
                        mean_nonak = (mean_nonak_raw - mean_nonak_raw.min()) / (mean_nonak_raw.max() - mean_nonak_raw.min())

                        # --- Hitung rata-rata dalam cluster ---
                        mean_akademik_fix = mean_akademik.mean()
                        mean_nonak_fix = mean_nonak.mean()

                        # --- Tentukan prioritas utama ---
                        selisih = mean_akademik_fix - mean_nonak_fix

                        threshold_selisih = 0.13
                        threshold_ak_tinggi = 0.5
                        threshold_nonak_tinggi = 0.45

                        if selisih > threshold_selisih:
                            kecendrungan = "Akademik"
                        elif selisih < -threshold_selisih:
                            kecendrungan = "Non-Akademik"
                        else:
                            # dua-duanya relatif seimbang
                            if mean_akademik_fix > threshold_ak_tinggi and mean_nonak_fix > threshold_nonak_tinggi:
                                kecendrungan = "Seimbang (Aktif)"
                            elif mean_akademik_fix < threshold_ak_tinggi or mean_nonak_fix < threshold_nonak_tinggi:
                                kecendrungan = "Seimbang (Pasif)"
                            else:
                                kecendrungan = "Netral"
                        

                        # Pilih kolom utama untuk aktivitas dominan
                        kolom_aktivitas = {
                            "Berapa jam rata-rata per minggu kamu gunakan untuk belajar mandiri (di luar kelas)?": "Belajar Mandiri",
                            "Seberapa sering kamu mengikuti kegiatan akademik tambahan (kuliah tamu, seminar, workshop)?": "Kegiatan Akademik Tambahan",
                            "Apakah kamu aktif mengikuti organisasi/UKM di kampus?": "Keaktifan Organisasi",
                            "Berapa jam rata-rata per minggu untuk organisasi/UKM?": "Waktu Organisasi",
                            "Berapa jam rata-rata per minggu untuk pekerjaan/hobi/olahraga?": "Waktu Hobi/Pekerjaan"
                        }

                        # Hitung rata-rata untuk tiap aktivitas
                        detail_aspek = {}
                        for col, label in kolom_aktivitas.items():
                            if col in df_cluster.columns:
                                detail_aspek[label] = df_cluster[col].mean()

                        # --- Ambil hingga 3 aktivitas dominan ---
                        if detail_aspek:
                            # Urutkan dari yang tertinggi ke terendah
                            top_aktivitas = sorted(detail_aspek.items(), key=lambda x: x[1], reverse=True)[:3]
                            aktivitas_dominan_list = [a[0] for a in top_aktivitas]
                            aktivitas_dominan_str = ", ".join(aktivitas_dominan_list)
                        else:
                            aktivitas_dominan_list = []
                            aktivitas_dominan_str = "Tidak terdeteksi"

                        # --- Cek keselarasan antara prioritas dan aktivitas dominan ---
                        selaras = False
                        sama = False
                        if prioritas == "Akademik" and kecendrungan == "Akademik":
                            selaras = True
                        elif prioritas == "Non-Akademik" and kecendrungan == "Non-Akademik":
                            selaras = True
                        elif kecendrungan == "Seimbang (Aktif)" or kecendrungan == "Seimbang (Pasif)": 
                            sama = True
                            selaras = True

                        if sama:
                            warna = "❓"
                        else: warna = "🟢" if selaras else "🔴"
                    
                        hasil_results[cluster_id] = {
                                "kecendrungan": kecendrungan,
                                "mean_ak" : mean_akademik_fix,
                                "mean_nonak": mean_nonak_fix,
                                "prioritas" : prioritas,
                                "dominan" : aktivitas_dominan_str,
                                "selaras" : selaras,
                                "warna" : warna
                        }
                    
                    desc_dict = {
                        "Akademik": "🎓 Fokus pada kegiatan akademik, belajar mandiri, dan peningkatan IPK.",
                        "Non-Akademik": "🎭 Lebih aktif di kegiatan organisasi, UKM, atau pekerjaan luar kampus.",
                        "Seimbang (Aktif)": "⚖️ Aktif di dua sisi — akademik dan non-akademik berjalan beriringan.",
                        "Seimbang (Pasif)": "🌙 Kegiatan relatif seimbang tapi intensitasnya tidak terlalu tinggi.",
                        "Netral": "💤 Tidak menunjukkan kecenderungan yang kuat ke salah satu bidang."
                    }

                    rows = []
                    for cluster_id, count in cluster_counts.items():
                        hasil = hasil_results.get(cluster_id, {})
                        kec = hasil.get("kecendrungan", "Tidak diketahui")
                        
                        descriptions = desc_dict.get(kec, "Deskripsi tidak tersedia.")
                        
                        rows.append({
                            "Cluster": int(cluster_id),
                            "Jumlah": int(count),
                            "Kecenderungan": kec,
                            "Deskripsi": descriptions
                        })
                    df_desc = pd.DataFrame(rows).sort_values("Cluster")
                    st.markdown("---")
                    st.dataframe(df_desc, use_container_width=True)
                    st.info(f"Total data: **{len(df_km)}**")

                    # === Tambahan: Analisis Detail Tiap Cluster (Prioritas dan Kegiatan Dominan) ===
                    st.markdown("### 🔍 Analisis Detail Tiap Cluster (Prioritas dan Kegiatan Dominan)")

                    for cluster_id in sorted(cluster_counts.index):
                        df_cluster = df_km[df_km["cluster"] == cluster_id]
                        hasil = hasil_results.get(cluster_id, {})
                        kec = hasil.get("kecendrungan", "Tidak diketahui")
                        ak = hasil.get("mean_ak", "Tidak diketahui")
                        nonak = hasil.get("mean_nonak", "Tidak diketahui")
                        pri = hasil.get("prioritas", "Tidak diketahui")
                        dom = hasil.get("dominan", "Tidak diketahui")
                        slr = hasil.get("selaras", "Tidak diketahui")
                        warna = hasil.get("warna", "-")
                        # --- Tampilkan hasil ---
                        st.markdown(f"""
                            **Cluster {cluster_id}**
                            - 🎓 Rata-rata Akademik: `{ak:.2f}`
                            - 👥 Rata-rata Non-Akademik: `{nonak:.2f}`
                            - 🎯 Prioritas: **{pri}**
                            - ⚖️ Kecendrungan: **{kec}**
                            - ⌛ Aktivitas Dominan (Top 3): **{dom}**
                            - {warna} Keselarasan: **{'Selaras' if slr else 'Tidak Selaras'}**
                            """)

                    st.markdown("---")

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

            df_numeric = df_for_cluster.select_dtypes(include=["number"]).copy()

            db_kdist, db_run, db_scatter, db_dist, db_hasil = st.tabs([
                "📊 K-Distance Plot", "▶️ Run", "🟢 Scatter (PCA 2D)", "📦 Distribusi", "📌 Hasil DBSCAN"
            ])

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

            with db_kdist:
                if df_numeric.empty:
                    st.error("❌ Tidak ada kolom numerik yang bisa digunakan untuk DBSCAN!")
                else:
                    M_k = _prepare_matrix("db_use_scaled_k", "db_use_pca_k", "db_ncomp_k")
                    k_value = st.slider("Pilih nilai k (biasanya = min_samples)", min_value=3, max_value=50, value=6, key="k_db")
                    if st.button("🔍 Tampilkan K-Distance Plot", key="btn_kdist"):
                        with st.spinner("Menghitung k-distance..."):
                            fig, _ = plot_k_distance_graph_manual(M_k, k=k_value)
                            st.pyplot(fig)
                            st.info("💡 Cari 'titik siku' (elbow). Nilai Y di titik siku ≈ **eps optimal**.\n"
                                    "   Tips: samakan k dengan `min_samples` yang akan kamu pakai di tab Run.")

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

            with db_scatter:
                if all(k in st.session_state for k in ["db_labels", "db_matrix_used"]):
                    fig_dbscan = visualize_dbscan_results(
                        st.session_state["db_matrix_used"],
                        st.session_state["db_labels"]
                    )
                    st.pyplot(fig_dbscan)
                    if "db_params" in st.session_state:
                        p = st.session_state["db_params"]
                        st.caption(f"Param: eps={p['eps']}, min_samples={p['min_samples']} • "
                                   f"Clusters={p['n_clusters']} • Noise={p['n_noise']} • Matrix={p['shape']}")
                else:
                    st.warning("Jalankan DBSCAN dulu di sub-tab **Run**.")

            with db_dist:
                if "db_df" in st.session_state and "cluster" in st.session_state["db_df"].columns:
                    cluster_counts = st.session_state["db_df"]["cluster"].value_counts().sort_index()
                    bar_pie_distribution(cluster_counts)
                else:
                    st.warning("Jalankan DBSCAN dulu di sub-tab **Run**.")

            with db_hasil:
                st.subheader("📌 Deskripsi Hasil DBSCAN")
                if "db_df" in st.session_state and "cluster" in st.session_state["db_df"].columns:
                    df_db = st.session_state["db_df"].copy()
                    counts = df_db["cluster"].value_counts().sort_index()
                    total = int(len(df_db))
                    rows = []
                    for cid, cnt in counts.items():
                        pct = (cnt / total) * 100 if total > 0 else 0.0
                        rows.append({
                            "Cluster": int(cid),
                            "Jumlah": int(cnt),
                            "Persentase (%)": round(pct, 2),
                            "Keterangan": "Noise" if cid == -1 else f"Cluster {int(cid)}"
                        })
                    st.dataframe(pd.DataFrame(rows).sort_values(["Cluster"]), use_container_width=True)

                    if all(k in st.session_state for k in ["db_labels", "db_matrix_used"]):
                        labels = st.session_state["db_labels"]
                        M_used = st.session_state["db_matrix_used"]
                        s_values, per_cluster, overall = silhouette_scores_manual(M_used, labels)

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
                            fig_sil, ax_sil = plt.subplots(figsize=(5, 3))
                            sns.barplot(data=df_sil, x="Cluster", y="Silhouette Mean", ax=ax_sil, palette="viridis")
                            ax_sil.set_ylim(-1, 1)
                            ax_sil.set_title("Mean Silhouette per Cluster (exclude noise)")
                            ax_sil.grid(True, axis='y', alpha=0.3)
                            st.pyplot(fig_sil)

                    # Penjelasan noise (tabel + poin)
                    render_noise_section(df_db, label_col="cluster")

                    if "db_params" in st.session_state:
                        p = st.session_state["db_params"]
                        st.info(
                            f"Parameter: eps = **{p['eps']}**, min_samples = **{p['min_samples']}** • "
                            f"Clusters = **{p['n_clusters']}**, Noise = **{p['n_noise']}** • "
                            f"Matrix = **{p['shape']}** • Total data = **{total}**"
                        )
                else:
                    st.warning("Jalankan DBSCAN dulu di sub-tab **Run**.")

if __name__ == "__main__":
    main()
