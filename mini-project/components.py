import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def render_noise_section(df_with_labels, label_col="cluster"):
    if df_with_labels is None or label_col not in df_with_labels.columns:
        return
    total_n = int(len(df_with_labels))
    n_noise = int((df_with_labels[label_col] == -1).sum())
    pct_noise = (n_noise / total_n * 100) if total_n > 0 else 0.0

    st.subheader("🧩 Ringkasan Noise (Outlier)")
    noise_table = pd.DataFrame([
        {"Aspek": "Total Data", "Keterangan": total_n},
        {"Aspek": "Jumlah Noise (-1)", "Keterangan": n_noise},
        {"Aspek": "Persentase Noise", "Keterangan": f"{pct_noise:.2f}%"},
        {"Aspek": "Definisi Singkat", "Keterangan": "Data yang tidak cukup mirip dengan kelompok manapun sehingga dianggap pencilan (outlier)."},
    ])
    st.table(noise_table)

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

def bar_pie_distribution(series_counts):
    col1, col2 = st.columns(2)
    with col1:
        fig_bar, ax1 = plt.subplots(figsize=(4, 3))
        sns.barplot(x=series_counts.index, y=series_counts.values, palette="viridis", ax=ax1)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Jumlah Data")
        ax1.set_title("Distribusi Cluster (Bar)")
        st.pyplot(fig_bar)
    with col2:
        fig_pie, ax2 = plt.subplots(figsize=(4, 3))
        ax2.pie(
            series_counts.values,
            labels=[f"Cluster {i}" for i in series_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("viridis", len(series_counts))
        )
        ax2.set_title("Distribusi Cluster (Pie)")
        st.pyplot(fig_pie)
