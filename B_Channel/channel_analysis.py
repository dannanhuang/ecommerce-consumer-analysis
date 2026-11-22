"""Member B: reproducible customer-level channel analysis.
Run: python channel_analysis.py --input data/customer_clean.csv --output reports/channel
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

KEY = "CustomerID"
CHANNEL = "Marketing_Channel"
FIRST = "First_Purchase_Channel"
CLV = "Customer_Lifetime_Value"
CAC = "Customer_Acquisition_Cost"
REQUIRED = [KEY, CHANNEL, FIRST, CLV, CAC]


def load_data(path):
    path = Path(path)
    df = pd.read_csv(path, dtype={KEY: "string"})
    absent = sorted(set(REQUIRED) - set(df.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}. Confirm the schema with A.")
    if df.empty:
        raise ValueError("The input file has no customer records.")
    df[KEY] = df[KEY].str.strip()
    if df[KEY].isna().any() or df[KEY].eq("").any():
        raise ValueError("Missing customer IDs: ask A to check them before analysis.")
    if df[KEY].duplicated().any():
        raise ValueError("Repeated customer IDs: confirm the row grain with A; do not silently deduplicate.")
    audit = {"input": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "rows": len(df), "unique_customers": int(df[KEY].nunique()),
             "missing_raw": {c: int(df[c].isna().sum()) for c in REQUIRED},
             "population": "One row per unique customer; all customers retained in counts.", "missing_customer_id_count": 0, "duplicate_customer_id_count": 0}
    for col in [CHANNEL, FIRST]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    for col in [CLV, CAC]:
        numeric = pd.to_numeric(df[col], errors="coerce").astype(float)
        valid = pd.Series(np.isfinite(numeric), index=df.index) & numeric.ge(0)
        audit[f"invalid_or_missing_{col}"] = int((~valid).sum())
        df[col] = numeric.where(valid)
    eligible = df[CAC].gt(0) & df[CLV].notna()
    df["Calculated_CLV_to_CAC"] = (df[CLV] / df[CAC]).where(eligible)
    df["Calculated_CLV_to_CAC"] = df["Calculated_CLV_to_CAC"].replace([np.inf, -np.inf], np.nan)
    audit["zero_CAC"] = int(df[CAC].eq(0).sum())
    audit["excluded_from_ratio"] = int(df["Calculated_CLV_to_CAC"].isna().sum())
    if "CLV_to_CAC_Ratio" in df:
        supplied = pd.to_numeric(df["CLV_to_CAC_Ratio"], errors="coerce")
        comparable = supplied.notna() & df["Calculated_CLV_to_CAC"].notna()
        audit["supplied_ratio_comparable_rows"] = int(comparable.sum())
        audit["supplied_ratio_difference_over_0_011"] = int(
            ((supplied - df["Calculated_CLV_to_CAC"]).abs().gt(.011) & comparable).sum())
    if {"Purchase_Date", "Acquisition_Date"}.issubset(df.columns):
        purchase = pd.to_datetime(df["Purchase_Date"], errors="coerce")
        acquisition = pd.to_datetime(df["Acquisition_Date"], errors="coerce")
        audit["purchase_before_acquisition"] = int(purchase.lt(acquisition).sum())
        audit["unparseable_or_missing_dates"] = int((purchase.isna() | acquisition.isna()).sum())
        audit["date_policy"] = "Flag only; no date-based filtering in this channel comparison. Confirm meanings with A."
    return df, audit


def summarize(df, groups=None):
    groups = groups or [CHANNEL]
    table = df.groupby(groups, dropna=False, observed=True).agg(
        customer_count=(KEY, "nunique"),
        mean_cac=(CAC, "mean"),
        median_clv=(CLV, "median"),
        median_clv_cac=("Calculated_CLV_to_CAC", "median"),
        valid_cac_n=(CAC, "count"),
        valid_clv_n=(CLV, "count"),
        valid_ratio_n=("Calculated_CLV_to_CAC", "count"),
    ).reset_index()
    if FIRST in groups:
        denominator = table.groupby(FIRST)["customer_count"].transform("sum")
    else:
        denominator = table["customer_count"].sum()
    table["customer_share_pct"] = table["customer_count"] / denominator * 100
    return table.sort_values("customer_count", ascending=False).reset_index(drop=True)


def describe(table):
    if table.empty:
        return "No customers match this selection."
    def leaders(metric, highest=True):
        values = table[metric].dropna()
        if values.empty:
            return "unavailable"
        target = values.max() if highest else values.min()
        names = table.loc[table[metric].eq(target), CHANNEL].astype(str).tolist()
        return ", ".join(names) + (f" ({target:,.0f})" if metric == "customer_count" else f" ({target:,.2f})")
    return (f"Largest customer count: {leaders('customer_count')}; "
            f"lowest mean CAC: {leaders('mean_cac', False)}; "
            f"highest median estimated CLV/CAC: {leaders('median_clv_cac')}. "
            "Customer-level comparison within the selected first-purchase scope.")


def executive_summary(df, overall, cross):
    def lead(table, metric, ascending=False):
        values = table[metric].dropna()
        if values.empty:
            return "Unavailable", float("nan")
        best = values.min() if ascending else values.max()
        return ", ".join(table.loc[table[metric].eq(best), CHANNEL]), float(best)
    volume, count = lead(overall, "customer_count")
    cost, mean_cost = lead(overall, "mean_cac", True)
    value, clv = lead(overall, "median_clv")
    efficiency, ratio = lead(overall, "median_clv_cac")
    share = count / len(df) * 100
    lines = ["# Executive Summary",
        f"**Acquisition scale.** {volume} leads customer volume with {count:,.0f} customers ({share:.1f}% of the dataset).",
        f"**Acquisition cost.** {cost} has the lowest mean CAC ({mean_cost:,.2f} source units).",
        f"**Estimated value.** {value} has the highest median estimated CLV ({clv:,.2f} source units); "
        f"{efficiency} has the highest median customer-level CLV/CAC ({ratio:,.2f}x).",
        "## First-purchase consistency"]
    winners = []
    for first, group in cross.groupby(FIRST, observed=True):
        winner, multiple = lead(group, "median_clv_cac")
        winners.append(winner)
        lines.append(f"- **{first}:** {winner} leads median CLV/CAC at {multiple:,.2f}x "
                     f"across a subgroup of {group.customer_count.sum():,.0f} customers.")
    consistency = ("The same CLV/CAC leader appears in every observed first-purchase subgroup."
                   if len(set(winners)) == 1 else
                   "The CLV/CAC leader changes across first-purchase subgroups; the overall leader does not represent every purchase context.")
    lines += [consistency,
        "## Decision implications",
        "Use separate lenses for acquisition scale, cost, and estimated value. Review channels within the relevant first-purchase context before prioritizing further investigation. "
        "These comparisons do not estimate incremental marketing returns or an optimal budget allocation.",
        "## Method and limitations",
        "One row per unique customer. Counts retain all records; financial metrics use valid non-negative values. "
        "CLV/CAC is calculated per customer with positive CAC before taking the group median. "
        "Valid sample counts accompany each metric. Cross-tab customer shares are normalized within each first-purchase subgroup. "
        "CLV is estimated, currency units are unverified, subgroup sizes differ, and channel differences are descriptive. "
        "Date anomalies are logged without removing records; no date-based or active-customer filter is applied."]
    return "\n\n".join(lines)


def run_analysis(input_path, output_dir):
    df, audit = load_data(input_path)
    overall = summarize(df)
    cross = summarize(df, [FIRST, CHANNEL])
    assert int(overall.customer_count.sum()) == len(df)
    assert int(cross.customer_count.sum()) == len(df)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    overall.to_csv(folder / "channel_summary.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(folder / "channel_by_first_purchase.csv", index=False, encoding="utf-8-sig")
    with (folder / "validation_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    report = executive_summary(df, overall, cross)
    report += "\n\n## Source record\n\nInput: `" + str(input_path) + "`\n\nSHA256: `" + audit["sha256"] + "`\n"
    report += "\nKaggle source: https://www.kaggle.com/datasets/rifatalam3/multi-channel-e-commerce-with-behavioral-metrics/data\n"
    (folder / "channel_report.md").write_text(report, encoding="utf-8")
    return df, overall, cross, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/customer_clean.csv")
    parser.add_argument("--output", default="reports/channel")
    args = parser.parse_args()
    _, table, _, _ = run_analysis(args.input, args.output)
    print(table.to_string(index=False))
    print(describe(table))
