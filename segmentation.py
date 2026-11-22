"""Train the reference segmentation or apply it to a cleaned customer batch."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "data/customer_clean.csv"
MODEL = ROOT / "models/customer_segments.json"
RESULTS = ROOT / "reports/segments"
FEATURES = ["Purchase_Frequency", "Days_Since_Last_Purchase", "Engagement_Score"]
OUTCOMES = ["Customer_Lifetime_Value", "Customer_Satisfaction", "Loyalty_Score"]
SEED = 42
CHOSEN_K = 4


def read_customers(path):
    frame = pd.read_csv(path, dtype={"CustomerID": "string"})
    required = ["CustomerID", *FEATURES]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame["CustomerID"] = frame["CustomerID"].str.strip()
    ids = frame["CustomerID"]
    if frame.empty or ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Supply one non-empty row per unique CustomerID.")
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[FEATURES]).all().all() or (frame[FEATURES[:2]] < 0).any().any():
        raise ValueError("Check missing or invalid behavior inputs before assigning segments.")
    return frame


def describe_centers(centers):
    # Give each of the four reviewed behavioral profiles a stable public ID.
    remaining = list(range(len(centers)))
    result = []
    for name, column, direction in [
        ("Frequent buyers", 0, 1),
        ("Long purchase gap", 1, 1),
        ("Lower engagement", 2, -1),
        ("Recent and engaged", 2, 1),
    ]:
        index = max(remaining, key=lambda i: direction * centers[i, column])
        remaining.remove(index)
        result.append({"id": len(result) + 1, "name": name, "center": centers[index].tolist()})
    return result


def train_reference(customers):
    scaler = StandardScaler()
    values = scaler.fit_transform(customers[FEATURES])
    sample = np.random.default_rng(SEED).choice(len(values), min(3000, len(values)), replace=False)
    evaluations = []
    for k in range(2, 9):
        estimator = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(values)
        sizes = np.bincount(estimator.labels_)
        evaluations.append({
            "k": k,
            "silhouette": float(silhouette_score(values[sample], estimator.labels_[sample])),
            "smallest_group": int(sizes.min()),
            "largest_group": int(sizes.max()),
        })
        if k == CHOSEN_K:
            segments = describe_centers(estimator.cluster_centers_)
    thresholds = {
        "frequent": float(customers[FEATURES[0]].quantile(.75)),
        "long_gap": float(customers[FEATURES[1]].quantile(.75)),
        "low_engagement": float(customers[FEATURES[2]].quantile(.25)),
    }
    return {
        "features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "segments": segments, "thresholds": thresholds, "reference_rows": len(values),
        "evaluation_sample": len(sample), "evaluation_seed": SEED, "evaluations": evaluations,
    }


def assign_segments(customers, model):
    # K-means prediction is the nearest saved center in standardized feature space.
    values = (customers[model["features"]].to_numpy() - model["mean"]) / model["scale"]
    segments = pd.DataFrame(model["segments"])
    centers = np.stack(segments["center"])
    distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    closest = segments.iloc[distances.argmin(axis=1)]
    scored = customers.copy()
    scored["segment_id"] = closest["id"].to_numpy()
    scored["customer_segment"] = closest["name"].to_numpy()
    limits = model["thresholds"]
    scored["rq2_match"] = (customers["Purchase_Frequency"] >= limits["frequent"]) & (
        (customers["Days_Since_Last_Purchase"] >= limits["long_gap"])
        | (customers["Engagement_Score"] <= limits["low_engagement"])
    )
    return scored


def summarize(scored, keys):
    return scored.groupby(keys, as_index=False).agg(
        customers=("CustomerID", "size"),
        median_frequency=("Purchase_Frequency", "median"),
        median_days=("Days_Since_Last_Purchase", "median"),
        median_engagement=("Engagement_Score", "median"),
        median_clv=("Customer_Lifetime_Value", "median"),
        median_satisfaction=("Customer_Satisfaction", "median"),
        median_loyalty=("Loyalty_Score", "median"),
        churned_customers=("Churned", "sum"),
        known_churn=("Churned", "count"),
        churn_share=("Churned", "mean"),
        rq2_customers=("rq2_match", "sum"),
    )


def write_results(customers, model, output):
    scored = assign_segments(customers, model)
    # New arrivals can be scored before outcomes are known.
    for column in [*OUTCOMES, "Churned"]:
        scored[column] = pd.to_numeric(scored.get(column, np.nan), errors="coerce")
    scored["Churned"] = scored["Churned"].where(scored["Churned"].isin([0, 1]))
    profiles = summarize(scored, ["segment_id", "customer_segment"])
    comparison = summarize(scored, ["rq2_match"])
    output.mkdir(parents=True, exist_ok=True)
    columns = ["CustomerID", "segment_id", "customer_segment", "rq2_match"]
    scored[columns].to_csv(output / "assignments.csv", index=False)
    profiles.to_csv(output / "profiles.csv", index=False)
    comparison.to_csv(output / "rq2_comparison.csv", index=False)
    print(f"Scored {len(scored):,} customers; {scored.rq2_match.sum():,} meet the RQ2 rule. {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("train", help="Rebuild the model and results from the full reference dataset")
    score = commands.add_parser("score", help="Apply the saved model to a batch or cumulative snapshot")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "train":
        customers = read_customers(REFERENCE)
        model = train_reference(customers)
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        MODEL.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
        write_results(customers, model, RESULTS)
    else:
        model = json.loads(MODEL.read_text(encoding="utf-8"))
        write_results(read_customers(args.input), model, args.output)


if __name__ == "__main__":
    main()
