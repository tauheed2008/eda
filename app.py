"""Retail Sales dashboard (Streamlit + Plotly).  Run:  streamlit run app.py"""
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import DEFAULT_INPUT, build_insights, calculate_kpis, load_data, run_cleaning

st.set_page_config(page_title="Retail Sales Dashboard", page_icon="\U0001F6CD", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18);
                          padding: 12px 14px; border-radius: 10px;}
</style>""", unsafe_allow_html=True)


def inr(x: float) -> str:
    """Format rupees compactly: 1.8 Cr, 4.2 L, or plain."""
    if abs(x) >= 1e7:
        return f"\u20b9{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"\u20b9{x / 1e5:.2f} L"
    return f"\u20b9{x:,.0f}"


@st.cache_data(show_spinner=False)
def prepare(file_bytes, drop_errors):
    raw = load_data(io.BytesIO(file_bytes)) if file_bytes else load_data(DEFAULT_INPUT)
    df, audit = run_cleaning(raw, drop_price_errors=drop_errors)
    return raw, df, audit


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload a retail CSV (optional)", type="csv")
    drop_errors = st.toggle("Treat extreme prices as errors", value=True,
                            help="Removes prices like 999,999 for a laptop or 1 for a mouse.")
try:
    raw, df, audit = prepare(upload.getvalue() if upload else None, drop_errors)
except Exception as exc:
    st.error(f"Could not process the data: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    f = df
    dates = df["Order_Date"].dropna()
    start, end = st.date_input("Order dates", (dates.min().date(), dates.max().date()),
                               min_value=dates.min().date(), max_value=dates.max().date())
    f = f[f["Order_Date"].dt.date.between(start, end)]
    for col in ["City", "Category", "Product", "Payment_Method", "Gender"]:
        picked = st.multiselect(col.replace("_", " "), sorted(df[col].dropna().unique()))
        if picked:
            f = f[f[col].isin(picked)]
    st.caption(f"{len(f):,} of {len(df):,} orders shown")

if f.empty:
    st.warning("No orders match these filters.")
    st.stop()

# ------------------------------------------------------------- header + KPIs
st.title("Retail Sales Dashboard")
k = calculate_kpis(f)
cols = st.columns(6)
cols[0].metric("Orders", f"{k['Orders']:,}")
cols[1].metric("Net sales", inr(k["Net sales (INR)"]))
cols[2].metric("After returns", inr(k["Net sales after returns (INR)"]))
cols[3].metric("Avg order value", inr(k["Average order value (INR)"]))
cols[4].metric("Return rate", f"{k['Return rate (%)']:.1f}%")
cols[5].metric("Avg rating", f"{k['Average rating (1-5)']:.2f} / 5")

with st.expander("Key findings", expanded=True):
    for line in build_insights(f):
        st.markdown(f"- {line}")

tab_sales, tab_prod, tab_cust, tab_ops, tab_data = st.tabs(
    ["Sales", "Products", "Customers", "Returns & delivery", "Data quality"])

# ------------------------------------------------------------- sales
with tab_sales:
    monthly = f.groupby("Order_Month")["Net_Sales"].sum().reset_index()
    st.plotly_chart(px.line(monthly, x="Order_Month", y="Net_Sales", markers=True,
                            title="Monthly net sales (INR)"), use_container_width=True)
    a, b = st.columns(2)
    city = f.groupby("City")["Net_Sales"].sum().sort_values().reset_index()
    a.plotly_chart(px.bar(city, x="Net_Sales", y="City", orientation="h",
                          title="Net sales by city"), use_container_width=True)
    pay = f.groupby("Payment_Method")["Net_Sales"].sum().reset_index()
    b.plotly_chart(px.pie(pay, names="Payment_Method", values="Net_Sales", hole=0.45,
                          title="Net sales by payment method"), use_container_width=True)

# ------------------------------------------------------------- products
with tab_prod:
    a, b = st.columns(2)
    cat = f.groupby("Category")["Net_Sales"].sum().reset_index()
    a.plotly_chart(px.pie(cat, names="Category", values="Net_Sales", hole=0.45,
                          title="Net sales share by category"), use_container_width=True)
    prod = f.groupby("Product").agg(Net_Sales=("Net_Sales", "sum"), Units=("Quantity", "sum"),
                                    Avg_Rating=("Customer_Rating", "mean"),
                                    Return_Rate=("Is_Returned", "mean")).reset_index()
    prod["Return_Rate"] = (prod["Return_Rate"] * 100).round(1)
    prod["Avg_Rating"] = prod["Avg_Rating"].round(2)
    b.plotly_chart(px.bar(prod.sort_values("Net_Sales"), x="Net_Sales", y="Product",
                          orientation="h", title="Net sales by product"), use_container_width=True)
    st.subheader("Product scorecard")
    st.dataframe(prod.sort_values("Net_Sales", ascending=False), hide_index=True,
                 use_container_width=True)

# ------------------------------------------------------------- customers
with tab_cust:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f.dropna(subset=["Customer_Age"]), x="Customer_Age", nbins=20,
                                color="Gender", title="Customer age"), use_container_width=True)
    rate = f["Customer_Rating"].dropna().value_counts().sort_index().reset_index()
    rate.columns = ["Rating", "Orders"]
    b.plotly_chart(px.bar(rate, x="Rating", y="Orders", title="Rating distribution"),
                   use_container_width=True)
    order = ["Under 18", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    ag = f.groupby("Age_Group")["Net_Sales"].sum().reindex(order).dropna().reset_index()
    st.plotly_chart(px.bar(ag, x="Age_Group", y="Net_Sales", title="Net sales by age group"),
                    use_container_width=True)

# ------------------------------------------------------------- returns & delivery
with tab_ops:
    a, b = st.columns(2)
    rr = (f[f["Returned"].notna()].groupby("Category")["Is_Returned"].mean() * 100).reset_index()
    rr.columns = ["Category", "Return rate (%)"]
    a.plotly_chart(px.bar(rr, x="Category", y="Return rate (%)", title="Return rate by category"),
                   use_container_width=True)
    b.plotly_chart(px.histogram(f, x="Delivery_Days", nbins=10, title="Delivery days"),
                   use_container_width=True)
    dr = f.groupby("Delivery_Days")["Customer_Rating"].mean().reset_index()
    st.plotly_chart(px.line(dr, x="Delivery_Days", y="Customer_Rating", markers=True,
                            title="Average rating by delivery days"), use_container_width=True)

# ------------------------------------------------------------- data quality
with tab_data:
    c = st.columns(4)
    c[0].metric("Rows in file", f"{audit['original_rows']:,}")
    c[1].metric("Rows after cleaning", f"{audit['cleaned_rows']:,}")
    c[2].metric("Duplicates removed", audit["duplicates_removed"])
    c[3].metric("Price outliers", audit["price_outliers"])
    st.write("**Invalid values set to missing**")
    st.json(audit["invalid_set_to_missing"])
    st.write("**Missing values after cleaning**")
    st.json(audit["missing_after_cleaning"])
    flagged = df[df["Price_Outlier"] | df["Quantity_Outlier"]][
        ["Order_ID", "Product", "Quantity", "Original_Unit_Price_INR", "Price_Outlier", "Quantity_Outlier"]]
    st.write("**Flagged outlier orders**")
    st.dataframe(flagged, hide_index=True, use_container_width=True)
    st.write("**Filtered data**")
    st.dataframe(f, hide_index=True, use_container_width=True, height=320)
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8-sig"),
                       "retail_filtered.csv", "text/csv")
