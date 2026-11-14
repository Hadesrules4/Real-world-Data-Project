from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA, RFM, METRICS = (ROOT / "data/processed/cleaned_transactions.csv",
                      ROOT / "data/processed/customer_rfm.csv",
                      ROOT / "outputs/metrics.json")
st.set_page_config(page_title="Retail Sales Analytics", page_icon="🛍️", layout="wide")
st.title("Retail Sales & Customer Analytics")
st.caption("UCI Online Retail | Historical transactions, 2010–2011")
if not DATA.exists() or not METRICS.exists():
    st.warning("Run the pipeline first: python -m src.analysis")
    st.code("pip install -r requirements.txt\npython -m src.analysis\nstreamlit run dashboard.py")
    st.stop()
df = pd.read_csv(DATA, parse_dates=["InvoiceDate"])
rfm = pd.read_csv(RFM)
with st.sidebar:
    countries = sorted(df.Country.dropna().unique())
    selected = st.multiselect("Countries", countries, default=countries)
    min_date, max_date = df.InvoiceDate.min().date(), df.InvoiceDate.max().date()
    dates = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
filtered = df[df.Country.isin(selected)]
if isinstance(dates, tuple) and len(dates) == 2:
    filtered = filtered[filtered.InvoiceDate.dt.date.between(dates[0], dates[1])]
if filtered.empty:
    st.info("No records match these filters.")
    st.stop()
revenue, orders, customers = filtered.Revenue.sum(), filtered.InvoiceNo.nunique(), filtered.CustomerID.nunique()
a, b, c, d = st.columns(4)
a.metric("Revenue", f"£{revenue:,.0f}")
b.metric("Orders", f"{orders:,}")
c.metric("Customers", f"{customers:,}")
d.metric("Average order value", f"£{revenue/orders:,.2f}" if orders else "£0")
left, right = st.columns(2)
monthly = filtered.set_index("InvoiceDate").resample("MS").Revenue.sum().reset_index()
left.plotly_chart(px.line(monthly, x="InvoiceDate", y="Revenue", markers=True, title="Monthly revenue"), use_container_width=True)
top = filtered.groupby("Description", as_index=False).Revenue.sum().nlargest(10, "Revenue")
right.plotly_chart(px.bar(top, x="Revenue", y="Description", orientation="h", title="Top products"), use_container_width=True)
left, right = st.columns(2)
countries = filtered.groupby("Country", as_index=False).Revenue.sum().nlargest(10, "Revenue")
left.plotly_chart(px.bar(countries, x="Country", y="Revenue", title="Top countries"), use_container_width=True)
segments = rfm.Segment.value_counts().rename_axis("Segment").reset_index(name="Customers")
right.plotly_chart(px.pie(segments, names="Segment", values="Customers", title="RFM customer segments"), use_container_width=True)
