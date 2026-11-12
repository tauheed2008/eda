"""Retail Sales pipeline: load -> clean -> flag outliers -> engineer -> KPIs -> insights.

Imported by app.py. Every function below is one step of the workflow, in order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "retail_sales.csv"

EXPECTED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_Age", "Gender", "City", "Product",
    "Category", "Quantity", "Unit_Price_INR", "Discount_Percent",
    "Payment_Method", "Customer_Rating", "Delivery_Days", "Returned",
]

# Valid ranges. Anything outside becomes missing (never guessed or imputed).
VALID_RANGES = {
    "Customer_Age": (1, 100),
    "Quantity": (1, None),
    "Unit_Price_INR": (1, None),
    "Discount_Percent": (0, 100),
    "Customer_Rating": (1, 5),
    "Delivery_Days": (1, None),
}

NULL_MARKERS = {"", "nan", "null", "none", "n/a", "na", "#name?"}
GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female",
              "fem ale": "Female", "other": "Other"}
PAYMENT_MAP = {"upi": "UPI", "credit card": "Credit Card", "debit card": "Debit Card",
               "net banking": "Net Banking", "cash": "Cash"}
YES_NO_MAP = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
              "no": "No", "n": "No", "false": "No", "0": "No"}


# ------------------------------------------------------------------ 1. LOAD
def load_data(source) -> pd.DataFrame:
    """Read a CSV path or file-like object and tidy the header names."""
    df = pd.read_csv(source)
    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


# ------------------------------------------------------------------ 2. CLEAN
def _normalise_text(series: pd.Series, mapping: dict | None = None,
                    title: bool = True) -> pd.Series:
    """Trim spaces, turn null markers into NA, standardise case, apply aliases."""
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.mask(s.str.lower().isin(NULL_MARKERS))
    lowered = s.str.lower()
    if mapping:
        return lowered.map(mapping).fillna(s.str.title() if title else s)
    return s.str.title() if title else s


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, audit). Invalid values become NA; nothing is imputed."""
    df = raw.copy()
    audit: dict = {"original_rows": len(df)}

    # Text columns
    df["Order_ID"] = df["Order_ID"].astype("string").str.strip()
    df["Gender"] = _normalise_text(df["Gender"], GENDER_MAP)
    df["Payment_Method"] = _normalise_text(df["Payment_Method"], PAYMENT_MAP)
    df["Returned"] = _normalise_text(df["Returned"], YES_NO_MAP)
    for col in ["City", "Product", "Category"]:
        df[col] = _normalise_text(df[col])

    # Types
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    for col in VALID_RANGES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exact duplicates
    audit["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    audit["duplicate_order_ids_left"] = int(df["Order_ID"].duplicated().sum())

    # Range validation
    invalid = {}
    for col, (low, high) in VALID_RANGES.items():
        bad = pd.Series(False, index=df.index)
        if low is not None:
            bad |= df[col] < low
        if high is not None:
            bad |= df[col] > high
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    audit["invalid_set_to_missing"] = invalid

    audit["cleaned_rows"] = len(df)
    return df, audit


# ------------------------------------------------------------ 3. OUTLIERS
def flag_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag prices that are unusual FOR THEIR OWN PRODUCT (IQR within product).

    A global IQR compares mice with laptops and flags dozens of normal rows.
    Judging each product against its own price band finds the real errors.
    """
    df = df.copy()
    grouped = df.groupby("Product")["Unit_Price_INR"]
    q1, q3 = grouped.transform(lambda s: s.quantile(0.25)), grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df["Price_Outlier"] = (df["Unit_Price_INR"] < q1 - 3 * iqr) | (df["Unit_Price_INR"] > q3 + 3 * iqr)

    q1q, q3q = df["Quantity"].quantile([0.25, 0.75])
    df["Quantity_Outlier"] = df["Quantity"] > q3q + 3 * (q3q - q1q)
    return df


# ------------------------------------------------------------ 4. FEATURES
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date parts, age groups and the sales money columns."""
    df = df.copy()
    df["Order_Month"] = df["Order_Date"].dt.to_period("M").astype("string")
    df["Order_Weekday"] = df["Order_Date"].dt.day_name()
    df["Age_Group"] = pd.cut(
        df["Customer_Age"], bins=[0, 17, 25, 35, 45, 55, 65, 100],
        labels=["Under 18", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
    ).astype("string")

    df["Gross_Sales"] = df["Quantity"] * df["Unit_Price_INR"]
    df["Discount_Amount"] = df["Gross_Sales"] * df["Discount_Percent"] / 100
    df["Net_Sales"] = df["Gross_Sales"] - df["Discount_Amount"]

    # A returned order counts as fully refunded (the data has no partial refunds).
    # Unknown return status stays unknown rather than being assumed "No".
    df["Is_Returned"] = df["Returned"].eq("Yes")
    df["Net_Sales_After_Returns"] = df["Net_Sales"].where(df["Returned"].eq("No"))
    df.loc[df["Is_Returned"], "Net_Sales_After_Returns"] = 0.0
    return df


def run_cleaning(raw: pd.DataFrame, drop_price_errors: bool = True) -> tuple[pd.DataFrame, dict]:
    """Clean, flag outliers and engineer features in one call (used by the app too).

    Price outliers (e.g. a 999,999 laptop or a 1-rupee mouse) are almost certainly
    data-entry errors and would distort every sales total, so by default their price
    is set to missing. Pass drop_price_errors=False to keep them. Quantity outliers
    are only flagged, because a bulk order is plausible.
    """
    df, audit = clean_data(raw)
    df = flag_price_outliers(df)
    df["Original_Unit_Price_INR"] = df["Unit_Price_INR"]
    audit["price_outliers"] = int(df["Price_Outlier"].sum())
    audit["quantity_outliers"] = int(df["Quantity_Outlier"].sum())
    audit["price_errors_set_to_missing"] = bool(drop_price_errors)
    if drop_price_errors:
        df.loc[df["Price_Outlier"], "Unit_Price_INR"] = np.nan
    df = engineer_features(df)
    audit["missing_after_cleaning"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    return df, audit


# ---------------------------------------------------------------- 5. KPIs
def calculate_kpis(df: pd.DataFrame) -> dict:
    """Headline metrics. Each ratio uses only rows where its inputs are known."""
    known_return = df["Returned"].notna()
    priced = df["Net_Sales"].notna()
    return {
        "Orders": int(df["Order_ID"].nunique()),
        "Units sold": int(df["Quantity"].sum()),
        "Gross sales (INR)": round(float(df["Gross_Sales"].sum()), 2),
        "Discounts given (INR)": round(float(df["Discount_Amount"].sum()), 2),
        "Net sales (INR)": round(float(df["Net_Sales"].sum()), 2),
        "Net sales after returns (INR)": round(float(df["Net_Sales_After_Returns"].sum()), 2),
        "Average order value (INR)": round(float(df.loc[priced, "Net_Sales"].mean()), 2) if priced.any() else 0.0,
        "Return rate (%)": round(float(df.loc[known_return, "Is_Returned"].mean() * 100), 2) if known_return.any() else 0.0,
        "Average rating (1-5)": round(float(df["Customer_Rating"].mean()), 2),
        "Average delivery (days)": round(float(df["Delivery_Days"].mean()), 2),
        "Average discount (%)": round(float(df["Discount_Percent"].mean()), 2),
    }


# --------------------------------------------------------------- 6. INSIGHTS
def build_insights(df: pd.DataFrame) -> list[str]:
    """Plain-English findings computed from the data (no hard-coded numbers)."""
    out: list[str] = []
    by_cat = df.groupby("Category")["Net_Sales"].sum().sort_values(ascending=False)
    out.append(f"{by_cat.index[0]} is the top category with {by_cat.iloc[0] / by_cat.sum():.0%} of net sales.")
    by_prod = df.groupby("Product")["Net_Sales"].sum().sort_values(ascending=False)
    out.append(f"{by_prod.index[0]} leads products; {by_prod.index[-1]} earns the least.")
    monthly = df.groupby("Order_Month")["Net_Sales"].sum()
    out.append(f"Best month is {monthly.idxmax()}; weakest is {monthly.idxmin()}.")
    ret = df[df["Returned"].notna()].groupby("Category")["Is_Returned"].mean().sort_values(ascending=False)
    out.append(f"{ret.index[0]} has the highest return rate at {ret.iloc[0]:.1%}.")
    rating = df["Customer_Rating"].mean()
    out.append(f"Average customer rating is {rating:.2f} out of 5, which is {'low' if rating < 3.5 else 'healthy'}.")
    city = df.groupby("City")["Net_Sales"].sum().sort_values(ascending=False)
    out.append(f"{city.index[0]} is the strongest city by net sales.")
    corr = df[["Delivery_Days", "Customer_Rating"]].corr().iloc[0, 1]
    out.append(f"Delivery time vs rating correlation is {corr:+.2f}, so delivery speed "
               f"{'does' if abs(corr) > 0.3 else 'does not clearly'} explain ratings.")
    return out
