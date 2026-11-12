"""Retail Sales EDA: reusable loading, cleaning, feature engineering and reporting.

Run from the project directory:
    python data_processing.py --input eda_practice_retail_sales.csv

The script preserves raw observations where possible. Invalid values are set to
missing and reported; IQR outliers are flagged, never automatically deleted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # Data cleaning/KPI functions remain usable without chart extras.
    plt = None
    sns = None

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "eda_practice_retail_sales.csv"

EXPECTED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_Age", "Gender", "City", "Product",
    "Category", "Quantity", "Unit_Price_INR", "Discount_Percent",
    "Payment_Method", "Customer_Rating", "Delivery_Days", "Returned",
]
NUMERIC_COLUMNS = [
    "Customer_Age", "Quantity", "Unit_Price_INR", "Discount_Percent",
    "Customer_Rating", "Delivery_Days",
]
TEXT_COLUMNS = [
    "Order_ID", "Gender", "City", "Product", "Category", "Payment_Method", "Returned",
]
OUTLIER_COLUMNS = ["Quantity", "Unit_Price_INR", "Gross_Sales", "Net_Order_Value", "Delivery_Days"]


def load_data(source: str | Path | Any) -> pd.DataFrame:
    """Load a CSV path or file-like object and normalize header whitespace."""
    df = pd.read_csv(source, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


def _clean_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    result = result.replace({
        "": pd.NA, "null": pd.NA, "NULL": pd.NA, "None": pd.NA,
        "none": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NAN": pd.NA,
        "#NAME?": pd.NA,
    })
    return result


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize capitalization and a small, explicit set of obvious aliases."""
    if "Gender" in df:
        gender = df["Gender"].astype("string").str.strip().str.lower()
        gender = gender.str.replace(r"\s+", " ", regex=True)
        gender = gender.replace({
            "female": "Female", "f": "Female", "fem ale": "Female",
            "male": "Male", "m": "Male", "other": "Other",
        })
        # Keep any unrecognized non-empty value visible for review.
        df["Gender"] = gender.str.title().replace({"Fem Ale": "Female"})
    for col in ["City", "Product", "Category", "Payment_Method"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
            # Title case is appropriate for this dataset's labels (e.g. Net Banking).
            df[col] = df[col].str.title()
    if "Payment_Method" in df:
        df["Payment_Method"] = df["Payment_Method"].replace({
            "Upi": "UPI", "Net Banking": "Net Banking", "Debit Card": "Debit Card",
            "Credit Card": "Credit Card",
        })
    if "Returned" in df:
        returned = df["Returned"].astype("string").str.strip().str.lower()
        df["Returned"] = returned.replace({
            "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
            "no": "No", "n": "No", "false": "No", "0": "No",
        })
    return df


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the retail dataset and return ``(cleaned_df, audit_report)``."""
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    original_rows, original_columns = len(df), len(df.columns)
    missing_required = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    # Normalize text/null markers before type conversion.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _clean_text(df[col])

    # Preserve identifiers as strings; do not convert them to numeric types.
    for col in ["Order_ID"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()

    # Dates are in ISO format in the supplied file. Coerce bad dates to missing.
    if "Order_Date" in df:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    # Parse numeric fields and record malformed entries.
    malformed_numeric: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        if col in df:
            before_nonnull = df[col].notna()
            parsed = pd.to_numeric(df[col], errors="coerce")
            malformed_numeric[col] = int((before_nonnull & parsed.isna()).sum())
            df[col] = parsed

    # Standardize recognizable text categories.
    df = _standardize_categories(df)

    # Remove exact duplicate records only. Distinct rows sharing an Order_ID are
    # retained and reported, because they could represent multiple line items.
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    remaining_duplicate_order_ids = 0
    if "Order_ID" in df:
        valid_ids = df.loc[df["Order_ID"].notna(), "Order_ID"]
        remaining_duplicate_order_ids = int(valid_ids.duplicated().sum())

    # Invalid values are set to NA, not imputed. Domain rules are deliberately
    # conservative so unusual legitimate orders are not discarded.
    invalid_counts: dict[str, int] = {}
    invalid_rules = {
        "Customer_Age": lambda s: (s < 0) | (s > 110),
        "Quantity": lambda s: s <= 0,
        "Unit_Price_INR": lambda s: s <= 0,
        "Discount_Percent": lambda s: (s < 0) | (s > 100),
        "Customer_Rating": lambda s: (s < 1) | (s > 5),
        "Delivery_Days": lambda s: s <= 0,
    }
    for col, rule in invalid_rules.items():
        if col in df:
            mask = df[col].notna() & rule(df[col])
            invalid_counts[col] = int(mask.sum())
            df.loc[mask, col] = np.nan

    # Date and category feature engineering.
    if "Order_Date" in df:
        df["Order_Year"] = df["Order_Date"].dt.year.astype("Int64")
        df["Order_Month"] = df["Order_Date"].dt.month.astype("Int64")
        df["Month_Name"] = df["Order_Date"].dt.month_name().astype("string")
        df["Order_Day"] = df["Order_Date"].dt.day_name().astype("string")
        df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype("string")

    if "Customer_Age" in df:
        df["Customer_Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=[-0.001, 17, 25, 35, 45, 55, 65, 110],
            labels=["Under 18", "18–25", "26–35", "36–45", "46–55", "56–65", "66+"],
        ).astype("string")

    # Sales calculations: order value is pre-return and excludes no records.
    required_sales = {"Quantity", "Unit_Price_INR"}
    if required_sales.issubset(df.columns):
        df["Gross_Sales"] = df["Quantity"] * df["Unit_Price_INR"]
    else:
        df["Gross_Sales"] = np.nan
    if "Discount_Percent" in df:
        df["Discount_Amount"] = df["Gross_Sales"] * df["Discount_Percent"] / 100
        df["Net_Order_Value"] = df["Gross_Sales"] - df["Discount_Amount"]
    else:
        df["Discount_Amount"] = np.nan
        df["Net_Order_Value"] = np.nan

    if "Returned" in df:
        df["Is_Returned"] = df["Returned"].eq("Yes")
        # Assumption: a returned order is treated as fully returned because the
        # source has no return quantity or refund amount. Unknown return status
        # remains unknown instead of being treated as a non-return.
        known_not_returned = df["Returned"].eq("No")
        df["Estimated_Net_Sales_After_Returns"] = df["Net_Order_Value"].where(known_not_returned, np.nan)
        df.loc[df["Is_Returned"], "Estimated_Net_Sales_After_Returns"] = 0
    else:
        df["Is_Returned"] = False
        df["Estimated_Net_Sales_After_Returns"] = df["Net_Order_Value"]

    # IQR flags are diagnostic only; they do not delete rows.
    outlier_counts: dict[str, int] = {}
    iqr_bounds: dict[str, dict[str, float | None]] = {}
    for col in OUTLIER_COLUMNS:
        if col not in df:
            continue
        valid = df[col].dropna()
        if len(valid) >= 4:
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
            df[f"{col}_Outlier_IQR"] = mask
            outlier_counts[col] = int(mask.sum())
            iqr_bounds[col] = {"lower": low, "upper": high}
        else:
            df[f"{col}_Outlier_IQR"] = False
            outlier_counts[col] = 0
            iqr_bounds[col] = {"lower": None, "upper": None}

    missing_report = {
        str(k): int(v) for k, v in df.isna().sum().items()
    }
    report: dict[str, Any] = {
        "original_rows": int(original_rows),
        "cleaned_rows": int(len(df)),
        "rows_removed_exact_duplicates": exact_duplicates,
        "remaining_duplicate_order_ids": remaining_duplicate_order_ids,
        "original_columns": int(original_columns),
        "cleaned_columns": int(len(df.columns)),
        "missing_expected_columns": missing_required,
        "malformed_numeric_values": malformed_numeric,
        "invalid_values_set_missing": invalid_counts,
        "iqr_outlier_counts": outlier_counts,
        "iqr_bounds": iqr_bounds,
        "missing_values_after_cleaning": missing_report,
    }
    return df, report


def calculate_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Calculate retail KPIs with explicit treatment of missing/returned values."""
    def series(col: str) -> pd.Series:
        if col not in df:
            return pd.Series(dtype="float64")
        return pd.to_numeric(df[col], errors="coerce")

    orders = int(df["Order_ID"].nunique()) if "Order_ID" in df else int(len(df))
    returned = df["Returned"].eq("Yes") if "Returned" in df else pd.Series(False, index=df.index)
    if "Returned" in df:
        known_return_status = df["Returned"].isin(["Yes", "No"])
        if "Order_ID" in df:
            known_return_orders = int(df.loc[known_return_status, "Order_ID"].nunique())
            returned_orders = int(df.loc[returned, "Order_ID"].nunique())
        else:
            known_return_orders = int(known_return_status.sum())
            returned_orders = int(returned.sum())
    else:
        known_return_orders = orders
        returned_orders = 0
    quantity = series("Quantity")
    gross = series("Gross_Sales")
    discount = series("Discount_Amount")
    net = series("Net_Order_Value")
    net_after_returns = series("Estimated_Net_Sales_After_Returns")
    rating = series("Customer_Rating")
    delivery = series("Delivery_Days")
    discount_pct = series("Discount_Percent")

    def total(s: pd.Series) -> float:
        return float(s.sum(min_count=1)) if s.notna().any() else 0.0

    def average(s: pd.Series) -> float:
        return float(s.mean()) if s.notna().any() else 0.0

    return {
        "Total Orders": orders,
        "Units Ordered": int(quantity.sum()) if quantity.notna().any() else 0,
        "Gross Sales (INR)": round(total(gross), 2),
        "Discount Amount (INR)": round(total(discount), 2),
        "Net Order Value (INR)": round(total(net), 2),
        "Estimated Sales After Returns (INR)": round(total(net_after_returns), 2),
        "Average Order Value (INR)": round(total(net) / orders, 2) if orders else 0.0,
        "Average Unit Price (INR)": round(average(series("Unit_Price_INR")), 2),
        "Returned Orders": returned_orders,
        "Return Rate (%)": round(returned_orders / known_return_orders * 100, 2) if known_return_orders else 0.0,
        "Average Customer Rating": round(average(rating), 2),
        "Average Delivery Days": round(average(delivery), 2),
        "Average Discount (%)": round(average(discount_pct), 2),
        "Units Not Returned": int(quantity.loc[df["Returned"].eq("No")].sum())
        if "Returned" in df and quantity.loc[df["Returned"].eq("No")].notna().any()
        else (int(quantity.sum()) if "Returned" not in df and quantity.notna().any() else 0),
    }


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Data_Type": df.dtypes.astype(str),
        "Missing_Count": df.isna().sum(),
        "Missing_Percent": (df.isna().mean() * 100).round(2),
        "Unique_Values": df.nunique(dropna=True),
    }).reset_index(names="Column")


def save_visualizations(df: pd.DataFrame, output_dir: str | Path) -> list[str]:
    """Save core EDA PNGs using Matplotlib/Seaborn when installed."""
    if plt is None or sns is None:
        return []
    sns.set_theme(style="whitegrid")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def finish(name: str) -> None:
        plt.tight_layout()
        path = out / name
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(str(path))

    if "Year_Month" in df and "Net_Order_Value" in df:
        monthly = df.groupby("Year_Month", dropna=True)["Net_Order_Value"].sum().reset_index()
        if not monthly.empty:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=monthly, x="Year_Month", y="Net_Order_Value", marker="o")
            plt.title("Monthly Net Order Value")
            plt.xlabel("Month")
            plt.ylabel("Net order value (INR)")
            plt.xticks(rotation=35, ha="right")
            finish("monthly_sales_trend.png")

    if {"Category", "Net_Order_Value"}.issubset(df.columns):
        category = df.groupby("Category", dropna=False)["Net_Order_Value"].sum().sort_values(ascending=False).reset_index()
        plt.figure(figsize=(9, 5))
        plt.bar(category["Category"].astype(str), category["Net_Order_Value"])
        plt.title("Net Order Value by Category")
        plt.ylabel("Net order value (INR)")
        finish("sales_by_category.png")

    if {"Product", "Net_Order_Value"}.issubset(df.columns):
        product = df.groupby("Product", dropna=False)["Net_Order_Value"].sum().sort_values(ascending=True).reset_index()
        plt.figure(figsize=(9, 5))
        plt.barh(product["Product"].astype(str), product["Net_Order_Value"])
        plt.title("Net Order Value by Product")
        plt.xlabel("Net order value (INR)")
        finish("sales_by_product.png")

    if "Customer_Age" in df:
        plt.figure(figsize=(9, 5))
        sns.histplot(data=df, x="Customer_Age", bins=20, kde=True, color="steelblue")
        plt.title("Customer Age Distribution")
        finish("customer_age_distribution.png")

    if "Customer_Rating" in df:
        plt.figure(figsize=(8, 5))
        rating_counts = df["Customer_Rating"].dropna().value_counts().sort_index()
        plt.bar(rating_counts.index.astype(str), rating_counts.values)
        plt.title("Customer Rating Distribution")
        finish("customer_rating_distribution.png")

    if {"Quantity", "Unit_Price_INR"}.issubset(df.columns):
        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=df, x="Quantity", y="Unit_Price_INR", hue="Category" if "Category" in df else None, palette="deep" if "Category" in df else None)
        plt.title("Quantity vs Unit Price")
        finish("quantity_vs_unit_price.png")

    if len(df.select_dtypes(include=np.number).columns) >= 2:
        numeric = df.select_dtypes(include=np.number).corr(numeric_only=True)
        plt.figure(figsize=(11, 8))
        sns.heatmap(numeric, cmap="coolwarm", center=0, annot=False)
        plt.title("Numeric Feature Correlation Heatmap")
        finish("correlation_heatmap.png")
    return saved


def run_pipeline(input_path: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = Path(output_dir)
    charts_dir = output / "charts"
    output.mkdir(parents=True, exist_ok=True)
    raw = load_data(input_path)
    cleaned, report = clean_data(raw)
    cleaned.to_csv(output / "retail_cleaned.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"KPI": k, "Value": v} for k, v in calculate_kpis(cleaned).items()]).to_csv(
        output / "kpi_summary.csv", index=False, encoding="utf-8-sig"
    )
    build_quality_report(cleaned).to_csv(output / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Column": k, "Outlier_Count": v,
                   "Lower_Bound": report["iqr_bounds"][k]["lower"],
                   "Upper_Bound": report["iqr_bounds"][k]["upper"]}
                  for k, v in report["iqr_outlier_counts"].items()]).to_csv(
        output / "outlier_report.csv", index=False, encoding="utf-8-sig"
    )
    (output / "cleaning_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    saved_charts = save_visualizations(cleaned, charts_dir)
    report["charts_saved"] = saved_charts
    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean retail sales data and create EDA outputs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to source retail CSV")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"), help="Output directory")
    args = parser.parse_args()
    cleaned, report = run_pipeline(args.input, args.output_dir)
    print("Retail EDA processing completed.")
    print(f"Rows: {report['original_rows']:,} -> {report['cleaned_rows']:,}")
    print(f"Exact duplicate rows removed: {report['rows_removed_exact_duplicates']:,}")
    print(f"Remaining duplicate Order_ID values: {report['remaining_duplicate_order_ids']:,}")
    print("\nKPIs:")
    for key, value in calculate_kpis(cleaned).items():
        print(f"  {key}: {value}")
    print(f"\nOutputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
