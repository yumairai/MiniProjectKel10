import numpy as np
from clustering_dbscan import euclidean_distance_matrix

def silhouette_scores_manual(X, labels):
    """
    Hitung silhouette per-sample (manual, Euclidean).
    Mengabaikan label -1 (noise).
    Return:
      s: array silhouette per sample (NaN untuk noise),
      per_cluster: dict {cluster_id: (mean, median, std, n)}
      overall_mean: rata-rata silhouette (tanpa noise & NaN)
    """
    labels = np.asarray(labels)
    n = X.shape[0]
    D = euclidean_distance_matrix(X)

    s = np.full(n, np.nan)
    clusters = [c for c in np.unique(labels) if c != -1]

    if len(clusters) < 1:
        return s, {}, np.nan

    for i in range(n):
        ci = labels[i]
        if ci == -1:
            continue

        same = np.where(labels == ci)[0]
        if len(same) <= 1:
            a_i = 0.0
        else:
            a_i = np.mean(D[i, same[same != i]])

        b_i = np.inf
        for cj in clusters:
            if cj == ci:
                continue
            other = np.where(labels == cj)[0]
            if len(other) == 0:
                continue
            b_i = min(b_i, np.mean(D[i, other]))

        if not np.isfinite(b_i):
            s[i] = 0.0
            continue

        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    per_cluster = {}
    for c in clusters:
        vals = s[labels == c]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            per_cluster[c] = (np.nan, np.nan, np.nan, 0)
        else:
            per_cluster[c] = (float(np.mean(vals)), float(np.median(vals)), float(np.std(vals)), int(len(vals)))

    overall_mean = float(np.nanmean(s))
    return s, per_cluster, overall_mean
