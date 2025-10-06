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
    """Parse rentang 'x - y' menjadi midpoint (float)."""
    s_str = s.astype(str)
    m = s_str.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    has_range = m[0].notna() & m[1].notna()
    midpoint = m[has_range].astype(float).mean(axis=1)
    out = s.copy()
    out.loc[has_range] = midpoint
    return out

# -----------------------------------------------
# ============== BASELINE ========================
# -----------------------------------------------

def manual_preprocess_baseline(data: pd.DataFrame):
    """PREPROCESSING BASELINE (versi stabil)."""
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
        "kadang (1 - 2 kali per bulan)": 2,
        "sering (1 - 2 kali per minggu)": 3,
        "setiap hari": 4,

        # Penting
        "sangat penting": 4, "penting": 3, "cukup penting": 2, "tidak terlalu penting": 1,

        # Boolean
        "ya": 1, "tidak": 0, "iya": 1, "nggak": 0,

        # Keseimbangan
        "sangat seimbang": 4, "seimbang": 3, "kurang seimbang": 2, "tidak seimbang": 1,
    }

    df = data.copy()

    # 1) Rapikan header & string
    df.columns = normalize_colnames(df.columns)
    for c in df.select_dtypes(include="object").columns:
        s = normalize_text_series(df[c])
        s = parse_range_midpoint(s)
        s = s.replace(mapping)
        s_num = pd.to_numeric(s, errors="coerce")
        df[c] = np.where(s_num.notna(), s_num, s)

    # 2) Ambil numerik saja
    df_num = df.select_dtypes(include=["number"]).copy()

    # 3) Imputasi median
    for col in df_num.columns:
        df_num[col] = df_num[col].fillna(df_num[col].median())

    # 4) Outlier IQR — DROP baris
    Q1, Q3 = df_num.quantile(0.25), df_num.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = ~((df_num < lower) | (df_num > upper)).any(axis=1)
    df_filtered = df_num[mask]

    # 5) Log-transform jika |skew|>1
    X_work = df_filtered.copy()
    for col in X_work.columns:
        sk = X_work[col].skew()
        if np.isfinite(sk) and abs(sk) > 1.0:
            shift = max(0, -X_work[col].min())
            X_work[col] = np.log1p(X_work[col] + shift)

    # 6) Z-score scaling
    mean_vals = X_work.mean()
    std_vals = X_work.std(ddof=0).replace(0, 1.0)
    X_scaled = (X_work - mean_vals) / std_vals

    report = {
        "mode": "baseline",
        "rows_before": len(df_num),
        "rows_after": len(df_filtered),
        "outlier_removed_rows": len(df_num) - len(df_filtered),
        "n_features": X_scaled.shape[1],
    }
    return df_filtered, X_scaled, report

# -----------------------------------------------
# ============== IMPROVED v2 =====================
# -----------------------------------------------

def manual_preprocess_v2(
    data: pd.DataFrame,
    drop_cols=("timestamp","waktu","date","tanggal","email","nama","name","id"),
    outlier="clip",
    scale="robust",
    small_cat_max_card=12,
    skew_thresh=1.0
):
    """PREPROCESSING IMPROVED v2 (mapping kegiatan diperluas)."""
    df = data.copy()
    df.columns = normalize_colnames(df.columns)

    for c in df.select_dtypes("object").columns:
        df[c] = normalize_text_series(df[c])

    to_drop = [c for c in df.columns if any(k in c.lower() for k in drop_cols)]
    if to_drop:
        df = df.drop(columns=to_drop)

    base_map = {
        "kurang dari 1 jam": 0.5,
        "1 - 2 jam": 1.5, "1-2 jam": 1.5,
        "3 - 5 jam": 4.0, "3-5 jam": 4.0,
        "6 - 8 jam": 7.0, "6-8 jam": 7.0,
        "9 - 12 jam": 10.5, "9-12 jam": 10.5,
        "lebih dari 12 jam": 12.5,

        # Frekuensi kegiatan
        "jarang (1 - 2 kali per semester)": 1,
        "kadang (1 - 2 kali per bulan)": 2,
        "sering (1 - 2 kali per minggu)": 3,
        "setiap hari": 4,

        # Frekuensi umum
        "selalu": 4, "sering": 3, "kadang-kadang": 2, "jarang": 1, "tidak pernah": 0,

        # Penting
        "sangat penting": 4, "penting": 3, "cukup penting": 2, "tidak terlalu penting": 1,

        # Boolean
        "ya": 1, "tidak": 0, "iya": 1, "nggak": 0,

        # Keseimbangan
        "sangat seimbang": 4, "seimbang": 3, "kurang seimbang": 2, "tidak seimbang": 1,
    }

    for c in df.select_dtypes("object").columns:
        s = parse_range_midpoint(df[c])
        s = s.replace(base_map)
        s_num = pd.to_numeric(s, errors="coerce")
        df[c] = np.where(s_num.notna(), s_num, s)

    obj_cols_after = df.select_dtypes("object").columns.tolist()
    small_cats = [c for c in obj_cols_after if 1 < df[c].nunique() <= small_cat_max_card]
    if small_cats:
        df = pd.get_dummies(df, columns=small_cats, drop_first=False, dtype=float)

    df_num = df.select_dtypes("number").copy()
    df_num = df_num.fillna(df_num.median())

    Q1, Q3 = df_num.quantile(0.25), df_num.quantile(0.75)
    IQR = (Q3 - Q1).replace(0, np.nan)
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    if outlier == "clip":
        df_num = df_num.clip(lower=lower, upper=upper, axis=1)
    else:
        mask = ~((df_num.lt(lower)) | (df_num.gt(upper))).any(axis=1)
        df_num = df_num[mask]

    for c in df_num.columns:
        col = df_num[c]
        if col.min() >= 0 and abs(col.skew()) > skew_thresh:
            df_num[c] = np.log1p(col)

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
        "n_rows": X_scaled.shape[0],
        "n_features": X_scaled.shape[1],
        "outlier_strategy": outlier,
        "scale": scale,
    }
    return df_num, X_scaled, report
