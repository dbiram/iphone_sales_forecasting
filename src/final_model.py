import pandas as pd
import lightgbm as lgb
import joblib

from src.data_loader import load_raw_data
from src.feature_engineering import generate_all_features
from src.train_model import get_feature_target_split


def train_final_model(data_path: str, model_path: str = "models/final_lgbm_model.pkl"):
    """
    Trains LightGBM on the full dataset and saves the model.
    """
    print("🔍 Loading and preparing data...")
    df = load_raw_data(data_path)
    df = generate_all_features(df)

    # Drop rows where lag features are NaN (start of each series)
    df = df.dropna(subset=["retail_price", "price_discounted", "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_rolling_mean_4"])

    print(f"Data ready: {len(df)} rows for training.")

    # Prepare features and target
    X, y, categorical_features = get_feature_target_split(df)

    print("Training LightGBM on full dataset...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(
        X, y,
        categorical_feature=categorical_features
    )

    print(f"Saving trained model to {model_path}")
    joblib.dump(model, model_path)

    print("Final model training complete.")

    return model