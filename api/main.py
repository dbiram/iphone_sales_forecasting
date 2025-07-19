from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from src.feature_engineering import generate_all_features

app = FastAPI()

# Load trained model
model = joblib.load("models/final_lgbm_model.pkl")

class HistoricalInput(BaseModel):
    SEMAINE: str
    BASIC_MODEL_NAME: str
    DATE_LANCEMENT: str
    NOMBRE_CMD: float
    ANCIENNETE_MODELE: int
    PRIX_DE_DETAIL: float
    RABAIS: int
    NOMBRE_CLIENT_FIN_CONTRAT: float
    CROIS_BYOD: float
    SEMAINE_NUM: int
    Fct_CROIS: float

class ForecastFeatureInput(BaseModel):
    PRIX_DE_DETAIL: float
    RABAIS: int
    NOMBRE_CLIENT_FIN_CONTRAT: float
    CROIS_BYOD: float
    Fct_CROIS: float

class ForecastRequest(BaseModel):
    historical_data: list[HistoricalInput]
    forecast_features: list[ForecastFeatureInput]

# Replicates load_raw_data() behavior (adapted for API)
def preprocess_raw_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    df = df_raw.rename(columns={
        "SEMAINE": "week",
        "BASIC_MODEL_NAME": "model_name",
        "DATE_LANCEMENT": "launch_date",
        "NOMBRE_CMD": "sales",
        "ANCIENNETE_MODELE": "product_age",
        "PRIX_DE_DETAIL": "retail_price",
        "RABAIS": "discount",
        "NOMBRE_CLIENT_FIN_CONTRAT": "end_contract_count",
        "CROIS_BYOD": "byod_growth",
        "SEMAINE_NUM": "week_number",
        "Fct_CROIS": "growth_factor"
    })

    # Date conversion
    df["week"] = pd.to_datetime(df["week"], utc=True).dt.tz_localize(None)
    df["launch_date"] = pd.to_datetime(df["launch_date"], utc=True).dt.tz_localize(None)

    # Create product_id
    df["product_id"] = df["model_name"].astype(str) + "_" + df["launch_date"].dt.strftime("%Y-%m-%d")

    # Sort
    df = df.sort_values(["product_id", "week"]).reset_index(drop=True)

    return df

@app.post("/predict/")
def predict(request: ForecastRequest):
    data_dicts = [item.dict() for item in request.historical_data]
    df_raw = pd.DataFrame(data_dicts)
    df_hist = preprocess_raw_input(df_raw)

    if len(df_hist) < 5:
        return {"error": "Vous devez fournir au moins 5 semaines d'historique pour effectuer une prédiction."}

    predictions = []

    # Rolling forecast loop
    for step, future_features in enumerate(request.forecast_features):
        # Generate features from df_hist
        df_features = generate_all_features(df_hist)

        df_features = df_features.dropna(subset=[
            "retail_price",
            "price_discounted",
            "sales_lag_1",
            "sales_lag_2",
            "sales_lag_4",
            "sales_rolling_mean_4"
        ])

        if df_features.empty:
            return {"error": "Impossible de générer des features valides avec les données fournies."}

        X = df_features[model.feature_name_]
        y_pred = model.predict(X)[-1]  # Use latest row for next forecast

        predictions.append(float(y_pred))

        # Build synthetic next week for rolling forecast
        last_week = df_hist.iloc[-1]["week"]
        next_week = pd.to_datetime(last_week) + pd.Timedelta(weeks=1)

        last_row = df_hist.iloc[-1].copy()
        next_row = last_row.copy()

        next_row["week"] = next_week
        next_row["sales"] = y_pred  # Use predicted sales

        # Update user-supplied features
        next_row["retail_price"] = future_features.PRIX_DE_DETAIL
        next_row["discount"] = future_features.RABAIS
        next_row["end_contract_count"] = future_features.NOMBRE_CLIENT_FIN_CONTRAT
        next_row["byod_growth"] = future_features.CROIS_BYOD
        next_row["growth_factor"] = future_features.Fct_CROIS

        # Increment product age, week_number
        next_row["product_age"] += 1
        next_row["week_number"] = next_row["week"].isocalendar().week

        df_hist = pd.concat([df_hist, pd.DataFrame([next_row])], ignore_index=True)

    return {"predictions": predictions}
