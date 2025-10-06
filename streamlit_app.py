import streamlit as st
import csv
import math
import random

# --- Fungsi bantu sederhana ---
def mean(values):
    return sum(values) / len(values) if values else 0

def std(values):
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values)) if values else 0

def quantile(values, q):
    sorted_vals = sorted(values)
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[int(pos)]
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * (pos - lower)

# --- Deteksi dan penanganan outlier pakai IQR ---
def handle_outliers_iqr(data_dict):
    cleaned = {}
    for col, vals in data_dict.items():
        if all(isinstance(v, (int, float)) for v in vals):
            q1 = quantile(vals, 0.25)
            q3 = quantile(vals, 0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            cleaned[col] = [min(max(v, lower), upper) for v in vals]  # clamp outliers
        else:
            cleaned[col] = vals
    return cleaned

# --- Normalisasi manual (Z-score) ---
def normalize_manual(data_dict):
    normed = {}
    for col, vals in data_dict.items():
        if all(isinstance(v, (int, float)) for v in vals):
            m = mean(vals)
            s = std(vals)
            normed[col] = [(v - m) / s if s != 0 else 0 for v in vals]
        else:
            normed[col] = vals
    return normed

# --- KMeans++ Manual ---
def initialize_centroids_kmeans_pp(X, k):
    random.seed(42)
    centroids = [random.choice(X)]
    while len(centroids) < k:
        distances = []
        for x in X:
            d = min(sum((x[i] - c[i])**2 for i in range(len(x))) for c in centroids)
            distances.append(d)
        total = sum(distances)
        probs = [d / total for d in distances]
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                centroids.append(X[i])
                break
    return centroids

def kmeans_manual(X, k=4, max_iters=100, tol=1e-5):
    centroids = initialize_centroids_kmeans_pp(X, k)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [math.sqrt(sum((x[i]-c[i])**2 for i in range(len(x)))) for c in centroids]
            idx = distances.index(min(distances))
            clusters[idx].append(x)
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append([mean([x[i] for x in cluster]) for i in range(len(X[0]))])
            else:
                new_centroids.append(random.choice(X))
        dif = sum(math.sqrt(sum((new_centroids[i][j] - centroids[i][j])**2 for j in range(len(X[0])))) for i in range(k))
        centroids = new_centroids
        if dif < tol:
            break
    labels = []
    for x in X:
        distances = [math.sqrt(sum((x[i]-c[i])**2 for i in range(len(x)))) for c in centroids]
        labels.append(distances.index(min(distances)))
    return labels, centroids

# --- STREAMLIT APP ---
st.set_page_config(page_title="Survei Keseimbangan Aktivitas Mahasiswa", layout="wide")
st.title("📊 Survei Keseimbangan Aktivitas Mahasiswa")
st.caption("Analisis clustering untuk memahami keseimbangan kegiatan akademik & non-akademik mahasiswa")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file:
    data = list(csv.DictReader(uploaded_file.read().decode('utf-8').splitlines()))
    st.subheader("📂 Data yang Diunggah")
    st.dataframe(data)

    # mapping manual
    mapping = {
        'Kurang dari 1 jam': 0.5, '1 - 2 jam': 1.5, '3 - 5 jam': 4,
        '6 - 8 jam': 7, 'Lebih dari 12 jam': 12,
        'Selalu': 4, 'Sering': 3, 'Kadang-kadang': 2, 'Jarang': 1, 'Tidak pernah': 0,
        'Sangat penting': 4, 'Penting': 3, 'Cukup penting': 2, 'Tidak terlalu penting': 1,
        'Ya': 1, 'Tidak': 0,
        'Sangat seimbang': 4, 'Seimbang': 3, 'Kurang seimbang': 2, 'Tidak seimbang': 1
    }

    # ubah ke numeric
    data_dict = {key: [] for key in data[0].keys()}
    for row in data:
        for k, v in row.items():
            try:
                data_dict[k].append(float(v))
            except:
                data_dict[k].append(mapping.get(v, 0))

    # handle outliers + normalisasi
    cleaned_data = handle_outliers_iqr(data_dict)
    normalized_data = normalize_manual(cleaned_data)

    # ubah ke list of list
    X = list(zip(*[normalized_data[col] for col in normalized_data]))

    if st.button("🚀 Proses Clustering (Manual KMeans++)"):
        labels, centroids = kmeans_manual(X, k=4)

        # gabungkan hasil
        result = []
        keys = list(normalized_data.keys())
        for i in range(len(X)):
            entry = {keys[j]: round(X[i][j], 2) for j in range(len(keys))}
            entry["Cluster"] = labels[i]
            result.append(entry)

        st.subheader("📊 Hasil Clustering (2 Desimal)")
        st.dataframe(result, use_container_width=True, height=300)

        # deskripsi cluster
        desc = [
            "Cluster 0 – Academic-Oriented: Fokus belajar, jarang ikut organisasi.",
            "Cluster 1 – Balanced: Cukup aktif di akademik & non-akademik.",
            "Cluster 2 – Non Academic-Oriented: Aktif di UKM/organisasi, belajar minim.",
            "Cluster 3 – Busy All-Rounder: Aktif di akademik, organisasi, bahkan kerja part-time."
        ]
        counts = [labels.count(i) for i in range(4)]
        st.subheader("📌 Deskripsi Cluster")
        for i in range(4):
            st.write(f"**{desc[i]}** (Jumlah: {counts[i]} mahasiswa)")
