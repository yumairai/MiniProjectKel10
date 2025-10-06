# data_preprocessing.py
# ===========================================================================================
# Semua util untuk:
# - PREPROCESSING data survei (baseline & improved v2)
# - JARAK GOWER manual (tanpa library clustering)
# - CLUSTERING manual:
#     * Agglomerative (Average-Linkage)
#     * DBSCAN (precomputed distance)
# - UTIL plotting sederhana (k-distance, kurva merge)
# Catatan:
# - Tidak memakai scikit-learn / scipy / hdbscan untuk clustering.
# - Hanya membutuhkan: numpy, pandas, matplotlib (untuk plot)
# ===========================================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    """
    Parse rentang "x - y" menjadi midpoint (float).
    Contoh: "3 - 5 jam" -> 4.0
    """
    s_str = s.astype(str)
    m = s_str.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    has_range = m[0].notna() & m[1].notna()
    midpoint = m[has_range].astype(float).mean(axis=1)
    out = s.copy()
    out.loc[has_range] = midpoint
    return out

def manual_preprocess_baseline(data: pd.DataFrame):
    """
    PREPROCESSING BASELINE (disederhanakan & diperbaiki):
    - Normalisasi string & mapping ordinal/boolean umum -> angka
    - Ambil numerik saja
    - Imputasi median
    - IQR outlier filter (DROP baris)
    - Log1p bila |skew|>1 (aman untuk non-negatif)
    - Z-score standardisasi
    """
    mapping = {
        # Durasi (pakai midpoint)
        "kurang dari 1 jam": 0.5,
        "1 - 2 jam": 1.5, "1-2 jam": 1.5,
        "3 - 5 jam": 4.0, "3-5 jam": 4.0,
        "6 - 8 jam": 7.0, "6-8 jam": 7.0,
        "9 - 12 jam": 10.5, "9-12 jam": 10.5,
        "lebih dari 12 jam": 12.5,
        # Frekuensi
        "selalu": 4, "sering": 3, "kadang-kadang": 2, "jarang": 1, "tidak pernah": 0,
        "jarang (1 - 2 kali per semester)": 1,
        "kadang-kadang (1 - 2 kali per bulan)": 2,
        "sering (hampir tiap bulan)": 3,
        "sangat sering (lebih dari 1 kali per bulan)": 4,
        # Penting
        "sangat penting": 4, "penting": 3, "cukup penting": 2, "tidak terlalu penting": 1,
        # Boolean
        "ya": 1, "tidak": 0,
        # Keseimbangan
        "sangat seimbang": 4, "seimbang": 3, "kurang seimbang": 2, "tidak seimbang": 1,
    }

    df = data.copy()

    # 1) Rapikan header & string
    df.columns = normalize_colnames(df.columns)
    for c in df.select_dtypes(include="object").columns:
        s = normalize_text_series(df[c])
        s = parse_range_midpoint(s)     # "x - y" -> midpoint
        s = s.replace(mapping)          # map kata kunci -> angka
        s_num = pd.to_numeric(s, errors="coerce")
        df[c] = np.where(s_num.notna(), s_num, s)

    # 2) Ambil numerik saja (kategorikal non-mapped terbuang di baseline)
    df_num = df.select_dtypes(include=["number"]).copy()

    # 3) Imputasi median
    for col in df_num.columns:
        df_num[col] = df_num[col].fillna(df_num[col].median())

    # 4) Outlier IQR — DROP baris jika ada outlier di salah satu kolom
    Q1, Q3 = df_num.quantile(0.25), df_num.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    before = len(df_num)
    mask = ~((df_num < lower) | (df_num > upper)).any(axis=1)
    df_filtered = df_num[mask]

    # 5) Log-transform jika |skew|>1 (safe untuk non-negatif; shift jika perlu)
    X_work = df_filtered.copy()
    for col in X_work.columns:
        sk = X_work[col].skew()
        if np.isfinite(sk) and abs(sk) > 1.0:
            shift = 0
            if X_work[col].min() < 0:  # jaga-jaga
                shift = -X_work[col].min()
            X_work[col] = np.log1p(X_work[col] + shift)

    # 6) Standardisasi (z-score)
    mean_vals = X_work.mean()
    std_vals = X_work.std(ddof=0).replace(0, 1.0)
    X_scaled = (X_work - mean_vals) / std_vals

    report = {
        "mode": "baseline",
        "rows_before": int(before),
        "rows_after": int(len(df_filtered)),
        "outlier_removed_rows": int(before - len(df_filtered)),
        "n_features": int(X_scaled.shape[1]),
    }
    return df_filtered, X_scaled, report

def manual_preprocess_v2(
    data: pd.DataFrame,
    drop_cols=("timestamp","waktu","date","tanggal","email","nama","name","id"),
    outlier="clip",          # "clip" (winsorize by IQR) atau "drop"
    scale="robust",          # "standard" | "robust" | None
    small_cat_max_card=12,   # one-hot untuk kategori kecil
    skew_thresh=1.0
):
    """
    PREPROCESSING IMPROVED v2 (disarankan):
    - Buang kolom identitas/timestamp umum
    - Normalisasi string
    - Mapping ordinal/boolean umum
    - One-hot untuk kategori kecil (tetap numerik)
    - Imputasi median
    - Outlier handling per kolom (clip default; atau drop)
    - Log1p bila |skew|>skew_thresh (fitur non-negatif)
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

    # 3) Mapping ordinals/boolean umum
    base_map = {
        "kurang dari 1 jam": 0.5,
        "1 - 2 jam": 1.5, "1-2 jam": 1.5,
        "3 - 5 jam": 4.0, "3-5 jam": 4.0,
        "6 - 8 jam": 7.0, "6-8 jam": 7.0,
        "9 - 12 jam": 10.5, "9-12 jam": 10.5,
        "selalu": 4, "sering": 3, "kadang-kadang": 2, "jarang": 1, "tidak pernah": 0,
        "lebih dari 12 jam": 12.5,
        "jarang (1 - 2 kali per semester)": 1,
        "kadang-kadang (1 - 2 kali per bulan)": 2,
        "sering (hampir tiap bulan)": 3,
        "sangat sering (lebih dari 1 kali per bulan)": 4,
        "selalu": 4, "sering": 3, "kadang-kadang": 2, "jarang": 1, "tidak pernah": 0,
        "sangat penting": 4, "penting": 3, "cukup penting": 2, "tidak terlalu penting": 1,
        "ya": 1, "tidak": 0,
        "sangat seimbang": 4, "seimbang": 3, "kurang seimbang": 2, "tidak seimbang": 1,
    }

    # 4) Terapkan ke kolom object: parse rentang -> map -> cast angka bila bisa
    obj_cols = df.select_dtypes("object").columns.tolist()
    for c in obj_cols:
        s = df[c]
        s = parse_range_midpoint(s)
        s = s.replace(base_map)
        s_num = pd.to_numeric(s, errors="coerce")
        df[c] = np.where(s_num.notna(), s_num, s)

    # 5) One-hot kategori kecil yang masih object
    obj_cols_after = df.select_dtypes("object").columns.tolist()
    small_cats = []
    for c in obj_cols_after:
        uniq = df[c].dropna().unique()
        if 1 < len(uniq) <= small_cat_max_card:
            small_cats.append(c)
    if small_cats:
        df = pd.get_dummies(df, columns=small_cats, drop_first=False, dtype=float)

    # 6) Ambil numerik + imputasi median
    df_num = df.select_dtypes(include=["number"]).copy()
    df_num = df_num.fillna(df_num.median(numeric_only=True))

    # 7) Outlier handling (clip/winsorize per kolom, atau drop baris)
    Q1, Q3 = df_num.quantile(0.25), df_num.quantile(0.75)
    IQR = (Q3 - Q1).replace(0, np.nan)         # hindari 0-division
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    if outlier == "clip":
        # Winsorize/clip tiap kolom -> lebih aman untuk survei (tidak buang responden)
        df_num = df_num.clip(lower=lower, upper=upper, axis=1)
    else:
        # Drop baris yang mengandung outlier di kolom manapun
        before = len(df_num)
        mask = ~((df_num.lt(lower)) | (df_num.gt(upper))).any(axis=1)
        df_num = df_num[mask]

    # 8) Log-transform jika miring & non-negatif (gunakan skewness sebenarnya)
    for c in df_num.columns:
        col = df_num[c]
        if col.min() >= 0:
            sk = col.skew()
            if np.isfinite(sk) and abs(sk) > skew_thresh:
                df_num[c] = np.log1p(col)

    # 9) Scaling (robust/standard/none)
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

# -----------------------------------------------
# ============== DISTANCE & METRICS ==============
# -----------------------------------------------

def gower_distance(df: pd.DataFrame) -> np.ndarray:
    """
    Matriks jarak Gower manual untuk data numerik/biner:
    - Numerik: dinormalisasi by range (max-min) per kolom.
    - Biner/one-hot: tetap 0/1; jarak = |x - y|.
    - Fitur konstan (range=0) dibobot 0 (tidak berkontribusi).
    """
    X = df.to_numpy(dtype=float)     # (n, m)
    n, m = X.shape

    # min, max, range per kolom
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    ranges = col_max - col_min

    # bobot: 1 utk fitur yg punya variasi, 0 utk konstan
    weights = np.where(ranges > 0, 1.0, 0.0)
    denom = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0

    # normalisasi numerik by range (fitur range=0 dibiarkan, karena bobotnya 0)
    safe_range = np.where(ranges == 0, 1.0, ranges)
    X_norm = np.where(ranges > 0, (X - col_min) / safe_range, X)

    # siapkan matriks jarak
    D = np.zeros((n, n), dtype=float)

    # hitung baris per baris
    for i in range(n):
        # beda absolut antara baris i dan semua baris -> (n, m)
        diff = np.abs(X_norm - X_norm[i, :])
        # beri bobot per fitur & agregasi ke jarak (n,)
        row = np.sum(diff * weights, axis=1) / denom
        D[i, :] = row

    # pastikan diagonal nol & simetris (secara teori sudah)
    np.fill_diagonal(D, 0.0)
    return D


def silhouette_precomputed(D: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette berbasis jarak precomputed (tanpa scikit-learn).
    """
    labels = np.asarray(labels)
    n = len(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")

    clusters = {c: np.where(labels == c)[0] for c in uniq}
    sils = []
    for i in range(n):
        ci = labels[i]
        same = clusters[ci]
        if len(same) <= 1:
            continue  # silhouette tidak terdefinisi untuk singleton
        a_i = float(np.mean(D[i, same[same != i]]))  # rata2 jarak intra-cluster

        b_i = float("inf")
        for cj, idxs in clusters.items():
            if cj == ci:
                continue
            b_i = min(b_i, float(np.mean(D[i, idxs])))

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        sils.append(s_i)

    return float(np.mean(sils)) if sils else float("nan")

# -----------------------------------------------
# ============== AGGLOMERATIVE MANUAL ============
# -----------------------------------------------

def _avg_link_dist(D: np.ndarray, cluster_a: np.ndarray, cluster_b: np.ndarray) -> float:
    """Jarak average-linkage antara dua cluster (mean jarak semua pasangan)."""
    sub = D[np.ix_(cluster_a, cluster_b)]
    return float(np.mean(sub)) if sub.size > 0 else 0.0

def agglomerative_average(D: np.ndarray, k: int):
    """
    Agglomerative clustering manual (average-linkage):
    - Mulai dari n cluster (singleton), gabung dua cluster terdekat sampai sisa k.
    - Kembalikan labels dan histori merge (untuk analisis lompatan jarak).
    Kompleksitas O(n^3) — cukup untuk n puluhan/ratusan.
    """
    n = D.shape[0]
    clusters = {i: np.array([i], dtype=int) for i in range(n)}
    next_id = n
    merges = []

    while len(clusters) > k:
        keys = list(clusters.keys())
        best = (None, None, float("inf"))
        for ii in range(len(keys)):
            for jj in range(ii + 1, len(keys)):
                a, b = keys[ii], keys[jj]
                d = _avg_link_dist(D, clusters[a], clusters[b])
                if d < best[2]:
                    best = (a, b, d)

        a, b, dist = best
        new_members = np.concatenate([clusters[a], clusters[b]])
        merges.append((a, b, float(dist), int(len(new_members))))
        del clusters[a]; del clusters[b]
        clusters[next_id] = new_members
        next_id += 1

    final_keys = list(clusters.keys())
    labels = np.empty(n, dtype=int)
    for cid_new, key in enumerate(final_keys):
        labels[clusters[key]] = cid_new
    return labels, merges

# -----------------------------------------------
# ================== DBSCAN MANUAL ===============
# -----------------------------------------------

def dbscan_precomputed(D: np.ndarray, eps: float, min_samples: int):
    """
    DBSCAN manual di atas matriks jarak precomputed:
    - eps: ambang jarak tetangga
    - min_samples: jumlah tetangga (termasuk diri sendiri) agar titik jadi 'core'
    - return labels: -1 = noise, 0..C-1 = id cluster
    """
    n = D.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    neighbors = [np.where(D[i] <= eps)[0] for i in range(n)]

    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors[i]
        if len(neigh) < min_samples:
            continue  # bukan core
        labels[i] = cid
        queue = list(neigh)
        q = 0
        while q < len(queue):
            j = queue[q]
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors[j]
                if len(neigh_j) >= min_samples:
                    for t in neigh_j:
                        if t not in queue:
                            queue.append(t)
            if labels[j] == -1:
                labels[j] = cid
            q += 1
        cid += 1
    return labels

# -----------------------------------------------
# ===================== PLOTS ====================
# -----------------------------------------------

def k_distance_values(D: np.ndarray, k: int = 6) -> np.ndarray:
    """Ambil k-distance (jarak ke tetangga ke-k) yang sudah diurutkan menaik (untuk cari 'siku' eps)."""
    sorted_d = np.sort(D, axis=1)
    kth = sorted_d[:, min(k, D.shape[0]-1)]
    return np.sort(kth)

def plot_k_distance_curve(kdist_sorted: np.ndarray, k: int):
    """Plot k-distance curve sederhana (pakai di Streamlit dengan st.pyplot)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(kdist_sorted)), kdist_sorted)
    ax.set_title(f"{k}-distance plot (pilih eps ≈ titik siku)")
    ax.set_xlabel("Points (sorted)")
    ax.set_ylabel(f"{k}-NN distance")
    return fig

def plot_merge_distances(merges):
    """Plot sejarah jarak penggabungan pada agglomerative (indikasi lompatan besar untuk pilih k)."""
    if not merges:
        return None
    dists = [m[2] for m in merges]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(1, len(dists)+1), dists, marker="o")
    ax.set_title("Sejarah jarak penggabungan (average-linkage)")
    ax.set_xlabel("Step merge")
    ax.set_ylabel("Average-linkage distance")
    return fig

# ======== PCA 2D & Scatter Plot (tanpa sklearn) ========

def pca_2d(X: np.ndarray):
    """
    PCA 2D manual (pakai SVD) untuk visualisasi.
    - X: array (n_samples, n_features), akan dicenter terlebih dulu.
    - Return: coords (n_samples, 2)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X harus 2D")
    n, m = X.shape
    if m == 1:
        # kalau cuma 1 fitur, tambahkan kolom nol biar bisa diplot 2D
        X = np.hstack([X, np.zeros((n, 1))])
        m = 2
    # center
    Xc = X - np.nanmean(X, axis=0)
    # SVD: Xc = U S Vt
    # komponen utama = kolom V (baris di Vt)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # ambil 2 komponen pertama
    V2 = Vt[:2, :].T            # (m,2)
    coords = Xc @ V2            # (n,2)
    return coords

def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, title="Cluster plot (PCA 2D)"):
    """
    Buat scatter plot 2D hasil PCA manual.
    - X: data numerik (n_samples, n_features) — gunakan DF 'clean_imputed' (bukan jarak).
    - labels: array cluster (DBSCAN: -1 = noise)
    """
    import matplotlib.pyplot as plt
    coords = pca_2d(X)  # (n,2)
    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for lab in unique_labels:
        mask = (labels == lab)
        if lab == -1:
            # Noise (khusus DBSCAN)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=28, marker="x", label="noise (-1)", alpha=0.85)
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=28, label=f"cluster {lab}", alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return fig

