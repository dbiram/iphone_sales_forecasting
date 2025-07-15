import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE_LANCEMENT"])
    
    # Rename columns
    df = df.rename(columns={
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

    # Convert week to datetime
    df["week"] = pd.to_datetime(df["week"], utc=True).dt.tz_localize(None)
    df["launch_date"] = pd.to_datetime(df["launch_date"], utc=True).dt.tz_localize(None)

    # Create product_id as model_name + launch_date string
    df["product_id"] = (
        df["model_name"].astype(str) + "_" + df["launch_date"].dt.strftime("%Y-%m-%d")
    )

    # Sort by product and date
    df = df.sort_values(["product_id", "week"]).reset_index(drop=True)
    
    return df
