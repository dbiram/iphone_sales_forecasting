import pandas as pd
import numpy as np

def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["week"].dt.year
    df["month"] = df["week"].dt.month
    df["quarter"] = df["week"].dt.quarter
    df["weekofyear"] = df["week"].dt.isocalendar().week

    # Cyclical encoding
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    
    return df

def generate_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df["product_age_weeks"] = (
        (df["week"] - df["launch_date"]).dt.days // 7
    ).clip(lower=0)

    df["product_age_bucket"] = pd.cut(
        df["product_age_weeks"],
        bins=[-1, 7, 20, 52, np.inf],
        labels=["cold_start", "growing", "mature", "late"]
    )

    return df

def generate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing retail prices
    df["retail_price"] = df["retail_price"].ffill()

    # Discounted price (optional)
    df["price_discounted"] = df["retail_price"] * (1 - df["discount"])

    # Bucket retail prices
    df["retail_price_bucket"] = pd.cut(
        df["retail_price"],
        bins=[0, 500, 799, 999, np.inf],
        labels=["low", "mid", "high", "premium"]
    )

    return df

def generate_lag_features(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    df = df.sort_values(["product_id", "week"])

    for lag in [1, 2, 4]:
        df[f"sales_lag_{lag}"] = (
            df.groupby("product_id")["sales"]
            .shift(lag)
        )

    df["sales_rolling_mean_4"] = (
        df.groupby("product_id")["sales"]
        .shift(1)
        .rolling(window=4, min_periods=1)
        .mean()
    )

    return df

def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = generate_time_features(df)
    df = generate_product_features(df)
    df = generate_interactions(df)
    df = generate_lag_features(df)
    return df
