import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
import numpy as np
from math import sqrt

def get_feature_target_split(df: pd.DataFrame):
    # Target
    y = df["sales"]

    # Feature selection
    features = [
        "product_age_weeks", "week_sin", "week_cos",
        "discount", "price_discounted",
        "end_contract_count", "byod_growth",
        "growth_factor",
        "retail_price_bucket", "product_age_bucket",
        "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_rolling_mean_4"
    ]
    # Numerical + categorical handling
    X = df[features].copy()

    categorical_features = ["retail_price_bucket", "product_age_bucket"]

    return X, y, categorical_features

def train_lightgbm(X: pd.DataFrame, y: pd.Series, categorical_features: list):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(
    X_train, y_train,
    categorical_feature=categorical_features
    )
    y_pred = model.predict(X_test)

    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"[Lightgbm] MAPE: {mape:.2f}%")
    print(f"[XGBoostLightgbm] RMSE: {rmse:.2f}")

    return model

def train_xgboost(X, y, categorical_features=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"[XGBoost] MAPE: {mape:.2f}%")
    print(f"[XGBoost] RMSE: {rmse:.2f}")

    return model

def train_catboost(X, y, categorical_features=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42
    )

    model.fit(X_train, y_train, cat_features=categorical_features)

    y_pred = model.predict(X_test)

    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"[CatBoost] MAPE: {mape:.2f}%")
    print(f"[CatBoost] RMSE: {rmse:.2f}")

    return model

