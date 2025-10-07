# data_preprocessing.py
# ==========================================================
# Modul preprocessing survei mahasiswa — versi ringan
# Hanya berisi fungsi preprocessing (tanpa clustering/plot)
# ==========================================================

import re
import numpy as np
import pandas as pd


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

def manual_preprocess_v2(
    data: pd.DataFrame,
    drop_cols=("timestamp", "date", "tanggal", "email", "nama", "name", "semester"),
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
