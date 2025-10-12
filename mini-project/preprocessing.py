import re
import numpy as np
import pandas as pd

def manual_preprocess_v2(
    data: pd.DataFrame,
    drop_cols=("timestamp", "date", "email", "nama", "name", "semester"),
    outlier="clip",          # "clip" (winsorize by IQR) atau "drop"
    scale="robust",          # "standard" | "robust" | None
    small_cat_max_card=12,   # one-hot untuk kategori kecil
    skew_thresh=1.0
):
    import pandas as pd
    import numpy as np
    import re

    # --- Daftar kolom akademik & non-akademik ---
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

    df = data.copy()

    # --- Normalisasi header & teks ---
    def normalize_colnames(cols):
        return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]
    df.columns = normalize_colnames(df.columns)

    def normalize_text_series(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.lower()
        s = s.str.replace("\u2013", "-", regex=False)
        s = s.str.replace(r"\s+", " ", regex=True)
        return s
    for c in df.select_dtypes("object").columns:
        df[c] = normalize_text_series(df[c])

    # --- Drop kolom tidak relevan ---
    to_drop = [c for c in df.columns if any(k in c.lower() for k in drop_cols)]
    if to_drop:
        df = df.drop(columns=to_drop)

    # --- Mapping nilai ordinal, range, boolean ---
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

    def parse_range_midpoint(s: pd.Series) -> pd.Series:
        s_str = s.astype(str)
        m = s_str.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
        has_range = m[0].notna() & m[1].notna()
        midpoint = m[has_range].astype(float).mean(axis=1)
        out = s.copy()
        out.loc[has_range] = midpoint
        return out

    obj_cols = df.select_dtypes("object").columns.tolist()
    for c in obj_cols:
        s = parse_range_midpoint(df[c])
        s = s.replace(base_map)
        s_num = pd.to_numeric(s, errors="coerce")
        df[c] = np.where(s_num.notna(), s_num, s)

    # --- One-hot kategori kecil ---
    obj_cols_after = df.select_dtypes("object").columns.tolist()
    small_cats = []
    for c in obj_cols_after:
        uniq = df[c].dropna().unique()
        if 1 < len(uniq) <= small_cat_max_card:
            small_cats.append(c)
    if small_cats:
        df = pd.get_dummies(df, columns=small_cats, drop_first=False, dtype=float)

    # --- Imputasi dan outlier handling ---
    df_num = df.select_dtypes(include=["number"]).fillna(df.median(numeric_only=True))
    Q1, Q3 = df_num.quantile(0.25), df_num.quantile(0.75)
    IQR = (Q3 - Q1).replace(0, np.nan)
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    if outlier == "clip":
        df_num = df_num.clip(lower=lower, upper=upper, axis=1)
    elif outlier == "drop":
        mask = ~((df_num.lt(lower)) | (df_num.gt(upper))).any(axis=1)
        df_num = df_num[mask]

    # --- Log transform kolom miring ---
    for c in df_num.columns:
        col = df_num[c]
        if col.min() >= 0:
            sk = col.skew()
            if np.isfinite(sk) and abs(sk) > skew_thresh:
                df_num[c] = np.log1p(col)

    # --- Scaling ---
    if scale == "robust":
        med = df_num.median()
        iqr = (df_num.quantile(0.75) - df_num.quantile(0.25)).replace(0, 1.0)
        X_scaled = (df_num - med) / iqr
    elif scale == "standard":
        X_scaled = (df_num - df_num.mean()) / df_num.std(ddof=0).replace(0, 1.0)
    elif scale == "minmax":
        X_scaled = (df_num - df_num.min()) / (df_num.max() - df_num.min())
    else:
        X_scaled = df_num.copy()

    # --- Hitung rata-rata akademik & non-akademik ---
    akademik_exist = [c for c in akademik_cols if c in X_scaled.columns]
    nonak_exist = [c for c in nonak_cols if c in X_scaled.columns]

    mean_akademik = X_scaled[akademik_exist].mean(axis=1)
    mean_nonak = X_scaled[nonak_exist].mean(axis=1)

    # --- Balancing: beri bobot ekstra pada non-akademik ---
    mean_nonak *= 1.2

    # --- Tambahkan kolom mean ke dataframe ---
    X_scaled["mean_akademik"] = mean_akademik
    X_scaled["mean_nonak"] = mean_nonak

    report = {
        "mode": "v2_with_balance",
        "dropped_cols": to_drop,
        "onehot_cols": small_cats,
        "n_rows": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "outlier_strategy": outlier,
        "scale": scale,
        "akademik_used": akademik_exist,
        "nonak_used": nonak_exist,
    }

    return df_num, X_scaled, report
