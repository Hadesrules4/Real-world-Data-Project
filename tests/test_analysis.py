import pandas as pd
from src.analysis import clean_data, build_rfm

def sample():
    return pd.DataFrame({
        "InvoiceNo": ["100", "100", "C101", "102", "103"],
        "StockCode": ["A", "B", "A", "C", "D"],
        "Description": ["MUG", "PLATE", "MUG", "TOY", None],
        "Quantity": [2, 1, 3, -1, 1],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"]),
        "UnitPrice": [5.0, 10.0, 5.0, 2.0, 1.0],
        "CustomerID": [123, 123, 456, 789, 111],
        "Country": ["UK", "UK", "UK", "France", "UK"]})

def test_cleaning_excludes_cancellations_and_invalid_rows():
    clean, quality = clean_data(sample())
    assert len(clean) == 2
    assert (clean.Revenue > 0).all()
    assert quality["cancellation_rows_detected"] == 1

def test_rfm_creates_customer_segments():
    clean, _ = clean_data(sample())
    result = build_rfm(clean)
    assert {"Recency", "Frequency", "Monetary", "Segment"}.issubset(result.columns)
    assert result.CustomerID.nunique() == len(result)
