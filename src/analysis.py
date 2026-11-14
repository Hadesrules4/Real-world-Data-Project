"""End-to-end UCI Online Retail analytics pipeline."""
from pathlib import Path
import json, zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, CHARTS, REPORTS = [ROOT / p for p in
    ("data/raw", "data/processed", "outputs/charts", "reports")]
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

def ensure_dirs():
    for p in (RAW, PROCESSED, CHARTS, REPORTS, ROOT / "outputs"):
        p.mkdir(parents=True, exist_ok=True)

def download_dataset():
    ensure_dirs()
    xlsx = RAW / "Online Retail.xlsx"
    if xlsx.exists():
        return xlsx
    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()
    archive_path = RAW / "online-retail.zip"
    archive_path.write_bytes(response.content)
    with zipfile.ZipFile(archive_path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
        xlsx.write_bytes(archive.read(name))
    archive_path.unlink()
    return xlsx

def clean_data(raw):
    df = raw.copy()
    rows = len(df)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ("Quantity", "UnitPrice", "CustomerID"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    cancellations = int(df["InvoiceNo"].str.upper().str.startswith("C").sum())
    df = df.dropna(subset=["InvoiceNo", "StockCode", "Description", "InvoiceDate",
                           "Quantity", "UnitPrice", "CustomerID", "Country"])
    df = df[(~df["InvoiceNo"].str.upper().str.startswith("C")) &
            (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    quality = {"raw_rows": rows, "duplicates_removed": duplicates,
               "cancellation_rows_detected": cancellations, "clean_rows": len(df)}
    return df, quality

def build_rfm(df):
    snapshot = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max().normalize()).days),
        Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum")).reset_index()
    rfm["R_score"] = (5 - np.ceil(rfm.Recency.rank(pct=True) * 4)).clip(1, 4).astype(int)
    rfm["F_score"] = np.ceil(rfm.Frequency.rank(pct=True) * 4).clip(1, 4).astype(int)
    rfm["M_score"] = np.ceil(rfm.Monetary.rank(pct=True) * 4).clip(1, 4).astype(int)
    def segment(r):
        if r.R_score >= 3 and r.F_score >= 3 and r.M_score >= 3: return "Champions"
        if r.R_score >= 3 and r.F_score >= 2: return "Loyal Customers"
        if r.R_score >= 3: return "Recent Customers"
        if r.F_score >= 3: return "At Risk"
        return "Needs Attention"
    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm

def forecast_daily_sales(df):
    daily = df.set_index("InvoiceDate").resample("D")["Revenue"].sum().asfreq("D", fill_value=0)
    frame = pd.DataFrame({"sales": daily})
    for lag in (1, 7, 14, 28): frame[f"lag_{lag}"] = frame.sales.shift(lag)
    frame["weekday"], frame["month"] = frame.index.dayofweek, frame.index.month
    frame = frame.dropna()
    if len(frame) < 60: return {"status": "skipped", "reason": "Not enough daily observations"}
    split = int(len(frame) * .8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    features = [c for c in frame.columns if c != "sales"]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(train[features], train.sales)
    pred = model.predict(test[features])
    return {"status": "completed", "test_rows": len(test),
            "random_forest_mae": float(mean_absolute_error(test.sales, pred)),
            "random_forest_rmse": float(np.sqrt(mean_squared_error(test.sales, pred))),
            "weekly_naive_mae": float(mean_absolute_error(test.sales, test.lag_7)),
            "test_start": str(test.index.min().date()), "test_end": str(test.index.max().date())}

def save_charts(df, rfm):
    monthly = df.groupby("YearMonth").Revenue.sum()
    fig, ax = plt.subplots(figsize=(11, 4)); monthly.plot(ax=ax, marker="o")
    ax.set(title="Monthly Revenue", xlabel="Month", ylabel="Revenue (£)")
    fig.tight_layout(); fig.savefig(CHARTS / "monthly_revenue.png", dpi=150); plt.close(fig)
    for filename, series, title, ylabel in [
        ("top_products.png", df.groupby("Description").Revenue.sum().nlargest(10).sort_values(), "Top Products by Revenue", "Product"),
        ("top_countries.png", df.groupby("Country").Revenue.sum().nlargest(10).sort_values(), "Top Countries by Revenue", "Country"),
        ("customer_segments.png", rfm.Segment.value_counts(), "Customer Segments", "Segment")]:
        fig, ax = plt.subplots(figsize=(9, 5)); series.plot(kind="barh", ax=ax)
        ax.set(title=title, xlabel="Revenue / Customers", ylabel=ylabel)
        fig.tight_layout(); fig.savefig(CHARTS / filename, dpi=150); plt.close(fig)

def run():
    ensure_dirs()
    raw = pd.read_excel(download_dataset())
    df, quality = clean_data(raw)
    if df.empty: raise ValueError("No valid positive sales rows remain after cleaning.")
    rfm = build_rfm(df)
    df.to_csv(PROCESSED / "cleaned_transactions.csv", index=False)
    rfm.to_csv(PROCESSED / "customer_rfm.csv", index=False)
    save_charts(df, rfm)
    monthly = df.groupby("YearMonth").Revenue.sum()
    country = df.groupby("Country").Revenue.sum()
    product = df.groupby("Description").Revenue.sum()
    metrics = {
        "data_quality": quality,
        "business_metrics": {
            "total_revenue_gbp": float(df.Revenue.sum()),
            "unique_customers": int(df.CustomerID.nunique()),
            "unique_invoices": int(df.InvoiceNo.nunique()),
            "unique_products": int(df.StockCode.nunique()),
            "average_order_value_gbp": float(df.groupby("InvoiceNo").Revenue.sum().mean()),
            "top_country": str(country.idxmax()), "top_product": str(product.idxmax()),
            "best_month": str(monthly.idxmax())},
        "customer_segments": {str(k): int(v) for k, v in rfm.Segment.value_counts().items()},
        "forecast": forecast_daily_sales(df)}
    (ROOT / "outputs/metrics.json").write_text(json.dumps(metrics, indent=2))
    b = metrics["business_metrics"]
    report = ["# Retail Sales & Customer Analytics — Results", "",
      f"- Revenue: £{b['total_revenue_gbp']:,.2f}",
      f"- Customers: {b['unique_customers']:,}", f"- Orders: {b['unique_invoices']:,}",
      f"- Products: {b['unique_products']:,}", f"- Best month: {b['best_month']}",
      f"- Top country: {b['top_country']}", f"- Top product: {b['top_product']}", "",
      "## Data quality", *[f"- {k}: {v}" for k, v in quality.items()], "",
      "## RFM segments", *[f"- {k}: {v}" for k, v in metrics["customer_segments"].items()], "",
      "## Forecast baseline", f"`{metrics['forecast']}`", "",
      "## Recommendations", "- Retain Champions and Loyal Customers with loyalty campaigns.",
      "- Test win-back campaigns for At Risk customers.", "- Monitor top-selling products and country concentration.",
      "- Treat the forecast as an educational baseline, not an operational prediction."]
    (REPORTS / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": run()

# CI trigger: generate real analysis outputs and dashboard screenshots.
