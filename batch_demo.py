import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "customer_selected_columns.csv"
DEMO = ROOT / "batch_demo"
PREPARED = DEMO / "prepared"    # Batches prepared for the simulation
INCOMING = DEMO / "incoming"    # Batches received so far
OUTPUT = DEMO / "customers_received.csv"


def prepare_batches():
    df = pd.read_csv(SOURCE, dtype={"CustomerID": "string"})
    dates = pd.to_datetime(df["Purchase_Date"], errors="coerce")

    # Group valid purchase dates into weeks from Monday to Sunday.
    weeks = dates.dt.to_period("W-SUN")
    first_two_weeks = sorted(weeks.dropna().unique())[:2]
    if len(first_two_weeks) < 2:
        raise ValueError("At least two weeks of purchase dates are required.")

    PREPARED.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)

    for number, week in enumerate(first_two_weeks, start=1):
        batch = df.loc[weeks == week]
        path = PREPARED / f"batch_{number:02d}.csv"
        batch.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{path.name} | week: {week} | customers: {len(batch)}")


def process_received_batches():
    files = sorted(INCOMING.glob("batch_*.csv"))
    if not files:
        raise SystemExit("No batch received. Copy a batch into batch_demo/incoming first.")

    batches = []
    for path in files:
        batch = pd.read_csv(path, dtype={"CustomerID": "string"})
        batches.append(batch)
        print(f"Received: {path.name} | rows: {len(batch)}")

    # Rebuild the cumulative table from all batches received so far.
    customers = pd.concat(batches, ignore_index=True)

    # Each customer has one record, so a repeated ID indicates a duplicate.
    if customers["CustomerID"].isna().any():
        raise ValueError("A received record has no CustomerID.")
    if customers["CustomerID"].duplicated().any():
        raise ValueError("A CustomerID appears more than once. Check the batch files.")

    customers.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Batches received: {len(files)}")
    print(f"Total customers: {len(customers)}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "process"])
    args = parser.parse_args()

    if args.action == "prepare":
        prepare_batches()
    else:
        process_received_batches()