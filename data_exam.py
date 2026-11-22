import argparse
from pathlib import Path

import pandas as pd


def examine_data(file_path):
    # 1. Read the file and preserve customer IDs as text
    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype={"CustomerID": "string"},
    )
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    print("\n1. DATASET OVERVIEW")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print(df.dtypes.to_string())

    # 2. Check missing values and duplicates
    print("\n2. MISSING VALUES AND DUPLICATES")
    print(df.isna().sum().to_string())

    customer_ids = df["CustomerID"].str.strip()
    id_counts = customer_ids.value_counts()

    print("Missing customer IDs:", customer_ids.isna().sum())
    print("Unique non-missing customer IDs:", len(id_counts))
    print("Customer IDs appearing more than once:", (id_counts > 1).sum())
    print("Fully duplicated rows beyond the first copy:", df.duplicated().sum())

    if (id_counts > 1).any():
        print("Examples of repeated IDs and their record counts:")
        print(id_counts[id_counts > 1].head().to_string())

    # 3. Inspect channel categories
    print("\n3. CHANNEL CATEGORIES")
    for col in ["Marketing_Channel", "First_Purchase_Channel"]:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False).to_string())

    # 4. Check dates
    print("\n4. DATE CHECKS")
    purchase = pd.to_datetime(
        df["Purchase_Date"], format="%Y-%m-%d", errors="coerce"
    )
    acquisition = pd.to_datetime(
        df["Acquisition_Date"], format="%Y-%m-%d", errors="coerce"
    )

    print("Purchase dates that could not be parsed:",
          (df["Purchase_Date"].notna() & purchase.isna()).sum())
    print("Acquisition dates that could not be parsed:",
          (df["Acquisition_Date"].notna() & acquisition.isna()).sum())
    print("Purchase date range:", purchase.min(), "to", purchase.max())

    both_dates_present = purchase.notna() & acquisition.notna()
    purchase_before_acquisition = both_dates_present & (purchase < acquisition)
    print("Rows with two valid dates:", both_dates_present.sum())
    print("Purchase before acquisition:", purchase_before_acquisition.sum())
    print(df.loc[purchase_before_acquisition,
                 ["CustomerID", "Purchase_Date", "Acquisition_Date"]]
            .head(3).to_string(index=False))

    # 5. Check numeric fields and their observed ranges
    print("\n5. NUMERIC FIELD CHECKS")
    numeric_cols = [
        "Customer_Lifetime_Value", "Customer_Acquisition_Cost",
        "Purchase_Frequency", "Days_Since_Last_Purchase", "Engagement_Score",
        "Customer_Satisfaction", "Loyalty_Score", "Churned",
        "Revenue", "Net_Revenue", "Discount_Applied", "Quantity",
        "Unit_Price", "CLV_to_CAC_Ratio",
    ]

    num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_summary = pd.DataFrame({
        "invalid_text": (df[numeric_cols].notna() & num.isna()).sum(),
        "min": num.min(),
        "median": num.median(),
        "max": num.max(),
    })
    print(numeric_summary.to_string())
    print("\nChurned label counts:")
    print(df["Churned"].value_counts(dropna=False).to_string())

    # 6. Check the inputs needed by members B and C
    print("\n6. CHANNEL AND CLUSTERING INPUTS")
    cac = num["Customer_Acquisition_Cost"]
    print("Zero acquisition cost:", cac.eq(0).sum())
    print("Negative acquisition cost:", cac.lt(0).sum())

    cluster_cols = [
        "Purchase_Frequency", "Days_Since_Last_Purchase", "Engagement_Score"
    ]
    print("Rows with all three numeric clustering inputs:",
          num[cluster_cols].notna().all(axis=1).sum())
    print("Negative purchase frequency:", num["Purchase_Frequency"].lt(0).sum())
    print("Negative days since last purchase:",
          num["Days_Since_Last_Purchase"].lt(0).sum())

    # 7. Compare related financial fields on comparable rows only
    print("\n7. FINANCIAL CONSISTENCY CHECKS")
    comparisons = [
        ("Revenue vs Quantity x Unit_Price",
         num["Revenue"], num["Quantity"] * num["Unit_Price"]),
        ("Net_Revenue vs Revenue x (1 - Discount_Applied)",
         num["Net_Revenue"], num["Revenue"] * (1 - num["Discount_Applied"])),
        ("Provided vs calculated CLV/CAC",
         num["CLV_to_CAC_Ratio"], num["Customer_Lifetime_Value"] / cac.where(cac > 0)),
    ]

    for name, reported, calculated in comparisons:
        valid = reported.notna() & calculated.notna()
        difference = (reported[valid] - calculated[valid]).abs()
        print(f"{name}: {(difference > 0.010001).sum()} differences "
              f"greater than 0.01 out of {valid.sum()} comparable rows")

def save_date_checked_data(input_path):
    # Read the 19-column project input
    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype={"CustomerID": "string"},
    )

    # Parse dates for comparison. Keep the original date columns in the file.
    purchase = pd.to_datetime(
        df["Purchase_Date"], format="%Y-%m-%d", errors="coerce"
    )
    acquisition = pd.to_datetime(
        df["Acquisition_Date"], format="%Y-%m-%d", errors="coerce"
    )

    # Classify every row, including rows where a date cannot be compared
    both_valid = purchase.notna() & acquisition.notna()
    before = both_valid & (purchase < acquisition)

    df["Date_Order_Status"] = "missing_or_invalid_date"
    df.loc[both_valid, "Date_Order_Status"] = "on_or_after_acquisition"
    df.loc[before, "Date_Order_Status"] = "purchase_before_acquisition"

    # Calculate elapsed days only when both dates are valid and in order
    elapsed_days = (purchase - acquisition).dt.days
    df["Days_From_Acquisition_To_Purchase"] = (
        elapsed_days.where(both_valid & ~before).astype("Int64")
    )

    output_path = Path(input_path).parent / "customer_clean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDATE CLEANING SUMMARY")
    print(df["Date_Order_Status"].value_counts().to_string())
    print("Rows saved:", len(df))
    print("Output file:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", nargs="?", type=Path,
        default=Path("data/customer_selected_columns.csv"),
    )
    args = parser.parse_args()
    examine_data(args.file)
    save_date_checked_data(args.file)