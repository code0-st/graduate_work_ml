import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_series(length=300, seasonality_period=50, amplitude=20, noise_level=5, base_level=50, seed=42):
    np.random.seed(seed)
    x = np.arange(length)
    seasonal_component = amplitude * np.sin(2 * np.pi * x / seasonality_period)
    noise = np.random.normal(0, noise_level, size=length)
    trend = 0.05 * x
    series = base_level + seasonal_component + noise + trend
    return series


def generate_mock_finam_file(length=300, start_date="2024-01-01", filename="mock_finam.csv"):
    series = generate_mock_series(length)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [start_dt + timedelta(days=i) for i in range(length)]

    data = []
    for i, date in enumerate(date_list):
        close = series[i]
        open_ = close + np.random.normal(0, 0.5)
        high = max(open_, close) + np.random.uniform(0, 1)
        low = min(open_, close) - np.random.uniform(0, 1)
        volume = np.random.randint(1000, 5000)
        data.append([
            date.strftime("%Y%m%d"),
            "100000",
            round(open_, 2),
            round(high, 2),
            round(low, 2),
            round(close, 2),
            volume
        ])

    df = pd.DataFrame(data, columns=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"])
    df.to_csv(filename, sep=";", index=False)
    print(f"Mock Finam file saved to {filename}")
    return df
