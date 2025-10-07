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
        """Fit K-Means clustering"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Inisialisasi centroid secara random
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]
        
        for i in range(self.max_iters):
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


def assign_cluster_labels(kmeans, feature_names):
    """
    Assign label ke setiap cluster berdasarkan karakteristik centroid
    Returns: dict mapping cluster_id -> (name, color, description)
    """
    cluster_profiles = {}
    
    for i in range(4):
        centroid = kmeans.centroids[i]
        
        # Hitung rata-rata nilai centroid
        avg_value = np.mean(centroid)
        
        # Identifikasi karakteristik dominan
        # Asumsi urutan fitur: belajar, organisasi, pekerjaan/hobi, ipk, tugas, keseimbangan
        study_high = centroid[0] > 0.6 if len(centroid) > 0 else False
        org_high = centroid[1] > 0.6 if len(centroid) > 1 else False
        ipk_high = centroid[3] > 0.6 if len(centroid) > 3 else False
        assignment_high = centroid[4] > 0.6 if len(centroid) > 4 else False
        
        # Logika penentuan cluster
        if study_high and ipk_high and not org_high:
            cluster_profiles[i] = {
                'name': '📚 Akademik Fokus',
                'color': '#4ECDC4',
                'description': 'Mahasiswa yang fokus pada prestasi akademik dengan prioritas tinggi pada belajar mandiri dan IPK'
            }
        elif org_high and avg_value > 0.5:
            cluster_profiles[i] = {
                'name': '⭐ Super Aktif',
                'color': '#FF6B6B',
                'description': 'Mahasiswa yang sangat aktif dalam berbagai kegiatan, baik akademik maupun non-akademik'
            }
        elif assignment_high and avg_value > 0.4 and avg_value < 0.7:
            cluster_profiles[i] = {
                'name': '⚖️ Balanced Achiever',
                'color': '#95E1D3',
                'description': 'Mahasiswa yang mampu menyeimbangkan berbagai aspek kehidupan kampus dengan baik'
            }
        else:
            cluster_profiles[i] = {
                'name': '😴 Struggling/Minimal Effort',
                'color': '#FFE66D',
                'description': 'Mahasiswa yang menghadapi kesulitan dalam mengelola aktivitas atau memberikan effort minimal'
            }
    
    # Pastikan semua kategori ada (jika ada yang duplikat, assign ulang)
    names_used = [v['name'] for v in cluster_profiles.values()]
    required_names = ['📚 Akademik Fokus', '⚖️ Balanced Achiever', '⭐ Super Aktif', '😴 Struggling/Minimal Effort']
    
    # Jika ada kategori yang belum terisi, assign ke cluster dengan centroid paling cocok
    for req_name in required_names:
        if req_name not in names_used:
            # Cari cluster yang belum punya nama unik
            for i in range(4):
                current_name = cluster_profiles[i]['name']
                if names_used.count(current_name) > 1:
                    if req_name == '📚 Akademik Fokus':
                        cluster_profiles[i] = {
                            'name': req_name,
                            'color': '#4ECDC4',
                            'description': 'Mahasiswa yang fokus pada prestasi akademik dengan prioritas tinggi pada belajar mandiri dan IPK'
                        }
                    elif req_name == '⚖️ Balanced Achiever':
                        cluster_profiles[i] = {
                            'name': req_name,
                            'color': '#95E1D3',
                            'description': 'Mahasiswa yang mampu menyeimbangkan berbagai aspek kehidupan kampus dengan baik'
                        }
                    elif req_name == '⭐ Super Aktif':
                        cluster_profiles[i] = {
                            'name': req_name,
                            'color': '#FF6B6B',
                            'description': 'Mahasiswa yang sangat aktif dalam berbagai kegiatan, baik akademik maupun non-akademik'
                        }
                    else:
                        cluster_profiles[i] = {
                            'name': req_name,
                            'color': '#FFE66D',
                            'description': 'Mahasiswa yang menghadapi kesulitan dalam mengelola aktivitas atau memberikan effort minimal'
                        }
                    break
    
    return cluster_profiles


def generate_cluster_insights(df_clustered, cluster_profiles, feature_names):
    """Generate insights untuk setiap cluster"""
    insights = {}
    
    for cluster_id in range(4):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id].drop('cluster', axis=1)
        profile = cluster_profiles[cluster_id]
        
        # Hitung statistik
        mean_values = cluster_data.select_dtypes(include=['number']).mean()
        
        # Generate insights
        characteristics = []
        recommendations = []
        
        # Analisis berdasarkan feature values
        for i, (feature, value) in enumerate(zip(feature_names, mean_values)):
            feature_lower = feature.lower()
            
            if 'belajar' in feature_lower and value > 0.6:
                characteristics.append("✅ Rajin belajar mandiri")
            elif 'belajar' in feature_lower and value < 0.3:
                characteristics.append("⚠️ Kurang aktif belajar mandiri")
                recommendations.append("Tingkatkan waktu belajar mandiri minimal 3-5 jam per minggu")
            
            if 'organisasi' in feature_lower or 'ukm' in feature_lower:
                if value > 0.6:
                    characteristics.append("✅ Sangat aktif berorganisasi")
                elif value < 0.2:
                    characteristics.append("ℹ️ Tidak/kurang aktif berorganisasi")
            
            if 'ipk' in feature_lower and value > 0.6:
                characteristics.append("✅ IPK menjadi prioritas utama")
            elif 'ipk' in feature_lower and value < 0.3:
                characteristics.append("⚠️ IPK bukan prioritas utama")
                recommendations.append("Pertimbangkan untuk lebih fokus pada pencapaian akademik")
            
            if 'tugas' in feature_lower and value > 0.6:
                characteristics.append("✅ Selalu mengerjakan tugas tepat waktu")
            elif 'tugas' in feature_lower and value < 0.3:
                characteristics.append("⚠️ Sering terlambat mengerjakan tugas")
                recommendations.append("Buat jadwal dan prioritaskan deadline tugas")
            
            if 'seimbang' in feature_lower and value < 0.3:
                characteristics.append("⚠️ Merasa hidup kurang seimbang")
                recommendations.append("Evaluasi pembagian waktu antara akademik dan non-akademik")
        
        # Default recommendations jika kosong
        if not recommendations:
            if '😴' in profile['name']:
                recommendations = [
                    "Buat jadwal harian yang terstruktur",
                    "Mulai dengan target kecil yang achievable",
                    "Cari study buddy atau join study group"
                ]
            elif '⭐' in profile['name']:
                recommendations = [
                    "Pastikan tidak overcommit pada terlalu banyak kegiatan",
                    "Prioritaskan kesehatan fisik dan mental",
                    "Tetap monitor IPK agar tidak terlalu drop"
                ]
            elif '📚' in profile['name']:
                recommendations = [
                    "Pertimbangkan untuk ikut organisasi untuk soft skill",
                    "Jaga work-life balance",
                    "Networking juga penting untuk karir"
                ]
            else:
                recommendations = [
                    "Pertahankan keseimbangan yang sudah baik",
                    "Terus konsisten dengan rutinitas",
                    "Eksplorasi peluang pengembangan diri"
                ]
        
        insights[cluster_id] = {
            'characteristics': characteristics,
            'recommendations': recommendations
        }
    
    return insights


# Streamlit App
def main():
    st.set_page_config(page_title="K-Means Clustering Mahasiswa", page_icon="🎓", layout="wide")
    
    st.title("🎓 Analisis Clustering Keseimbangan Aktivitas Mahasiswa")
    st.markdown("**Sistem Klasifikasi 4 Cluster: Akademik Fokus | Balanced Achiever | Super Aktif | Struggling/Minimal Effort**")
    st.markdown("---")
    
    # Upload file
    st.sidebar.header("📁 Upload Data Preprocessing")
    uploaded_file = st.sidebar.file_uploader("Upload file CSV hasil preprocessing", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data berhasil dimuat: {df.shape[0]} mahasiswa, {df.shape[1]} fitur")
        
        # Preview data
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))
            st.write(f"**Kolom:** {', '.join(df.columns.tolist())}")
        
        # Convert ke numpy
        X = df.values
        feature_names = df.columns.tolist()
        
        # Auto run clustering
        with st.spinner("🔄 Sedang melakukan clustering..."):
            # Run K-Means dengan 4 cluster
            kmeans = KMeans(k=4, max_iters=100, random_state=42)
            kmeans.fit(X)
            
            # Assign cluster labels
            cluster_profiles = assign_cluster_labels(kmeans, feature_names)
            
            # Tambahkan hasil ke dataframe
            df['cluster'] = kmeans.labels
            df['cluster_name'] = df['cluster'].map(lambda x: cluster_profiles[x]['name'])
            
        st.success("✅ Clustering selesai!")
        
        st.markdown("---")
        st.header("📊 Hasil Analisis Clustering")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Mahasiswa", len(df))
        with col2:
            st.metric("Jumlah Cluster", 4)
        with col3:
            inertia = kmeans.calculate_inertia(X)
            st.metric("Inertia (WCSS)", f"{inertia:.2f}")
        
        # Distribusi cluster
        st.markdown("---")
        st.subheader("📈 Distribusi Mahasiswa per Cluster")
        
        # Hitung distribusi berdasarkan cluster name
        cluster_dist = df.groupby('cluster').agg({
            'cluster_name': 'first',
            'cluster': 'count'
        }).rename(columns={'cluster': 'jumlah'})
        cluster_dist['persentase'] = (cluster_dist['jumlah'] / len(df) * 100).round(2)
        cluster_dist = cluster_dist.sort_values('jumlah', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(cluster_dist, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [cluster_profiles[i]['color'] for i in cluster_dist.index]
            bars = ax.bar(range(len(cluster_dist)), cluster_dist['jumlah'], 
                         color=colors, edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(len(cluster_dist)))
            ax.set_xticklabels([cluster_profiles[i]['name'] for i in cluster_dist.index], 
                              rotation=15, ha='right')
            ax.set_ylabel('Jumlah Mahasiswa', fontsize=12, fontweight='bold')
            ax.set_title('Distribusi Mahasiswa per Cluster', fontsize=14, fontweight='bold', pad=20)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Insights per cluster
        st.markdown("---")
        st.header("🔍 Profil & Rekomendasi per Cluster")
        
        insights = generate_cluster_insights(df, cluster_profiles, feature_names)
        
        # Urutkan berdasarkan jumlah mahasiswa
        sorted_clusters = cluster_dist.index.tolist()
        
        for cluster_id in sorted_clusters:
            profile = cluster_profiles[cluster_id]
            insight = insights[cluster_id]
            count = cluster_dist.loc[cluster_id, 'jumlah']
            pct = cluster_dist.loc[cluster_id, 'persentase']
            
            with st.expander(f"**{profile['name']}** — {count} mahasiswa ({pct}%)", expanded=True):
                st.markdown(f"**📝 Deskripsi:**")
                st.info(profile['description'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✨ Karakteristik:**")
                    for char in insight['characteristics']:
                        st.markdown(f"- {char}")
                
                with col2:
                    st.markdown("**💡 Rekomendasi:**")
                    for rec in insight['recommendations']:
                        st.markdown(f"- {rec}")
        
                # Heatmap
        st.markdown("---")
        st.subheader("🔥 Heatmap Karakteristik per Cluster")

        # Ukuran heatmap proporsional terhadap jumlah fitur
        fig, ax = plt.subplots(figsize=(2.2 * len(sorted_clusters), 0.6 * len(feature_names) + 3))

        centroids_sorted = kmeans.centroids[sorted_clusters]
        cluster_labels = [cluster_profiles[i]['name'] for i in sorted_clusters]

        sns.heatmap(
            centroids_sorted.T,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            xticklabels=cluster_labels,
            yticklabels=feature_names,
            cbar_kws={'label': 'Nilai (Normalized 0-1)'},
            linewidths=0.6,
            linecolor='gray',
            ax=ax
        )
        ax.set_title("Karakteristik Centroid per Cluster", fontsize=15, fontweight="bold", pad=15)
        ax.set_xlabel("Cluster", fontsize=12, fontweight="bold")
        ax.set_ylabel("Fitur", fontsize=12, fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        plt.yticks(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

        # =================================================================
        # Tambahan: Scatter Plot 2D antar cluster
        # =================================================================
        st.markdown("---")
        st.subheader("📉 Visualisasi Sebaran Cluster (2D Projection)")

        # Pastikan data numerik saja untuk visualisasi
        numeric_df = df.select_dtypes(include=['number']).copy()
        if 'cluster' not in numeric_df.columns:
            numeric_df['cluster'] = df['cluster']

        if numeric_df.shape[1] >= 3:
            x_feature = st.selectbox("Pilih sumbu X:", feature_names, index=0)
            y_feature = st.selectbox("Pilih sumbu Y:", feature_names, index=1)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            for cluster_id in sorted_clusters:
                cluster_points = df[df['cluster'] == cluster_id]
                ax2.scatter(
                    cluster_points[x_feature],
                    cluster_points[y_feature],
                    color=cluster_profiles[cluster_id]['color'],
                    label=cluster_profiles[cluster_id]['name'],
                    s=60,
                    edgecolor='black',
                    alpha=0.7
                )
            ax2.set_xlabel(x_feature, fontsize=12, fontweight="bold")
            ax2.set_ylabel(y_feature, fontsize=12, fontweight="bold")
            ax2.set_title("Sebaran Mahasiswa Berdasarkan Fitur", fontsize=14, fontweight="bold", pad=15)
            ax2.legend(title="Cluster", loc='best', fontsize=9)
            ax2.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("⚠️ Data tidak cukup fitur numerik untuk scatter plot 2D.")
        
        # Kesimpulan
        st.markdown("---")
        st.header("📌 Kesimpulan Keseluruhan")
        
        # Cluster terbesar
        largest_cluster = sorted_clusters[0]
        largest_name = cluster_profiles[largest_cluster]['name']
        largest_pct = cluster_dist.loc[largest_cluster, 'persentase']
        
        st.markdown(f"""
        ### Temuan Utama:
        
        1. **Cluster Dominan:** {largest_name} merupakan cluster terbesar dengan {largest_pct}% mahasiswa
        
        2. **Distribusi Keseimbangan:**
           - Mayoritas mahasiswa dapat dikategorikan ke dalam 4 tipe berbeda berdasarkan pola aktivitas mereka
           - Setiap cluster memiliki karakteristik dan kebutuhan yang unik
        
        3. **Rekomendasi Umum:**
           - **Akademik Fokus:** Perlu lebih banyak interaksi sosial dan pengalaman organisasi
           - **Balanced Achiever:** Pertahankan keseimbangan yang sudah baik
           - **Super Aktif:** Perhatikan kesehatan dan jangan sampai burnout
           - **Struggling/Minimal Effort:** Butuh pendampingan dan sistem support yang lebih baik
        """)
        
        # Download hasil
        st.markdown("---")
        st.subheader("💾 Download Hasil Clustering")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Clustering (CSV)",
            data=csv,
            file_name="hasil_clustering_mahasiswa.csv",
            mime="text/csv",
        )
    
    else:
        st.info("👆 Upload file CSV preprocessing di sidebar untuk memulai analisis")
        st.markdown("""
        ### 📋 Format File yang Dibutuhkan:
        
        File CSV dengan data yang sudah di-preprocessing dengan kolom:
        - **Belajar Mandiri** (normalized 0-1)
        - **Organisasi/UKM** (normalized 0-1)
        - **Pekerjaan/Hobi** (normalized 0-1)
        - **Prioritas IPK** (normalized 0-1)
        - **Tugas Tepat Waktu** (normalized 0-1)
        - **Keseimbangan Hidup** (normalized 0-1)
        
        #### Hasil yang akan didapat:
        - ✅ Klasifikasi otomatis ke 4 cluster
        - ✅ Karakteristik detail setiap cluster
        - ✅ Rekomendasi spesifik per cluster
        - ✅ Visualisasi lengkap
        - ✅ File hasil clustering untuk download
        """)


if __name__ == "__main__":
    main()