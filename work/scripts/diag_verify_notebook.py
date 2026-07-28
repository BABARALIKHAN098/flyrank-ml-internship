"""Exact replica of the w04 notebook code cell. Run top-to-bottom to verify outputs."""
from pathlib import Path
import numpy as np
import pandas as pd

# Walk up from this script until we find the repo root (has data/processed/refresh_feature_vector.csv)
HERE = Path(__file__).resolve()
for parent in [HERE, *HERE.parents]:
    if (parent / "data" / "processed" / "refresh_feature_vector.csv").exists():
        ROOT = parent
        break
SRC  = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
OUT  = ROOT / "work" / "outputs" / "baseline_action_score.csv"

df = pd.read_csv(SRC)
print(f"loaded {len(df):,} rows × {df.shape[1]} cols from {SRC}")
print(f"label base rate (random pick): {df['is_declining_label'].mean():.3f}")

stale_tier   = df["freshness_tier"].isin(["31-90", "91-180", "181+"]).astype(int)
visible      = (df["impressions_90d"] >= 500).astype(int)
slipping     = ((df["avg_position"] > 10) & (df["avg_position"] > 0)).astype(int)

stale_severity = np.clip(df["days_since_last_update"].fillna(0) / 365.0, 0, 1)
log_vis        = np.log1p(df["impressions_90d"].fillna(0))
pos_severity   = np.clip(1 - (df["avg_position"].fillna(50) / 50.0), 0, 1)

df["baseline_refresh_score"] = (
    stale_tier * stale_severity * (1 + log_vis / 10) * (1 + pos_severity)
    + visible   * 0.10
    + slipping  * 0.05
)

def reasons(row):
    codes = []
    if row["freshness_tier"] in ("31-90", "91-180", "181+") and row["days_since_last_update"] >= 30:
        codes.append("stale_over_30d")
    if row["impressions_90d"] >= 500:
        codes.append("still_visible_500imp")
    if row["avg_position"] > 10 and row["avg_position"] > 0:
        codes.append("position_slipping")
    if np.log1p(row["impressions_90d"]) >= 8:
        codes.append("high_traffic_log")
    if row["avg_position"] > 30:
        codes.append("deep_rank_gt_30")
    if row["clicks_90d"] == 0 and row["impressions_90d"] > 0:
        codes.append("no_clicks_yet")
    return ";".join(codes) if codes else "none"

df["reason_codes"] = df.apply(reasons, axis=1)
df = df.sort_values("baseline_refresh_score", ascending=False).reset_index(drop=True)
df["baseline_rank"] = np.arange(1, len(df) + 1)

def action(rank):
    if rank <= 50:   return "refresh_review_now"
    if rank <= 200:  return "refresh_review_next_sprint"
    if rank <= 1000: return "monitor"
    return "no_action"

df["suggested_action_baseline"] = df["baseline_rank"].apply(action)

labels = df["is_declining_label"].to_numpy()
scores = df["baseline_refresh_score"].to_numpy()
base   = labels.mean()

def precision_at_k(s, y, k):
    order = np.argsort(-np.asarray(s))
    return np.asarray(y)[order[:k]].mean()

print(f"\nbase rate (random pick): {base:.3f}")
print(f"{'k':>6}  {'precision@k':>12}  {'lift':>8}  {'cohort_label_rate':>18}")
for k in [50, 100, 200, 500, 1000, 5000]:
    p = precision_at_k(scores, labels, k)
    cohort_rate = labels[df["baseline_rank"].to_numpy() <= k].mean()
    print(f"{k:>6}  {p:>12.3f}  {p - base:>+8.3f}  {cohort_rate:>18.3f}")

print(f"\ntop score: {df['baseline_refresh_score'].max():.3f}   median: {df['baseline_refresh_score'].median():.3f}")
print()
print("=== top-20 for the hand review ===")
top20 = df.head(20)[["baseline_rank", "content_id", "client_id", "baseline_refresh_score",
                     "reason_codes", "impressions_90d", "clicks_90d", "avg_position",
                     "days_since_last_update", "is_declining_label", "trend_direction"]]
print(top20.to_string(index=False))
print()
print("=== reason-code frequency, top-50 ===")
top50 = df.head(50).copy()
top50["rc_list"] = top50["reason_codes"].str.split(";")
from collections import Counter
c = Counter([r for lst in top50["rc_list"] for r in lst])
print(c.most_common())