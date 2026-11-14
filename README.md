# Real-World Data Project: Retail Sales & Customer Analytics

An end-to-end Python data science project using the **UCI Online Retail** dataset. It covers data cleaning, exploratory data analysis, sales and product performance, RFM customer segmentation, a baseline daily-sales forecasting model, visualizations, and a Streamlit dashboard.

## Download

- [Complete project ZIP](Retail_Sales_Customer_Analytics_Project.zip)

## Dataset

The project downloads the original UCI Online Retail workbook automatically when the pipeline runs.

- Source: https://archive.ics.uci.edu/dataset/352/online+retail
- DOI: https://doi.org/10.24432/C5BW33
- License: CC BY 4.0
- Citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.

## Features

- Data cleaning: duplicates, missing values, cancellations, invalid quantities and prices
- Exploratory analysis of revenue, orders, customers, products, countries and time trends
- RFM customer segmentation (Recency, Frequency, Monetary)
- Daily sales forecasting baseline with chronological holdout evaluation
- Charts and generated Markdown report
- Interactive Streamlit dashboard
- Automated unit tests

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m src.analysis
streamlit run dashboard.py
```

Run tests with `pytest -q`.

## Important notes

The source dataset contains historical transactions from 2010–2011. Results are historical learning examples, not current market insights. The data pipeline generates cleaned data, charts, metrics, and the analysis report after the dataset has been downloaded. The ZIP includes all project source files and setup instructions; generated data outputs are produced at runtime.

## Technology stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Plotly, Excel, pytest.
