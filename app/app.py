import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Prévisions des Ventes iPhone", layout="centered")
st.title("📱 Prévisions des ventes iPhone (8 semaines maximum)")

st.header("1️⃣ Charger les données historiques")
st.write("Téléversez un fichier CSV contenant au minimum les **5 dernières semaines** avec les ventes réelles (`NOMBRE_CMD`).")

uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])

historical_data = None

if uploaded_file:
    historical_data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données historiques :")
    st.dataframe(historical_data)

# Input forecast window
st.header("2️⃣ Paramétrer les semaines futures")
prediction_window = st.number_input(
    "Combien de semaines futures souhaitez-vous prédire ?",
    min_value=1, max_value=8, value=4, step=1
)

st.write("Pour chaque semaine future, renseignez les paramètres commerciaux prévus :")

forecast_features = []

for week in range(prediction_window):
    st.subheader(f"🔮 Semaine future {week + 1}")

    prix = st.number_input(f"Prix de détail prévu (€) - semaine {week + 1}", min_value=500.0, value=1199.0)
    rabais = st.selectbox(f"Rabais prévu - semaine {week + 1}", options=[0, 1])
    clients_fin = st.number_input(f"Clients fin de contrat prévus - semaine {week + 1}", min_value=0, value=150)
    byod = st.number_input(f"Taux de croissance BYOD prévu - semaine {week + 1}", min_value=0.0, max_value=1.0, value=0.25)
    facteur_crois = st.number_input(f"Facteur de croissance prévu - semaine {week + 1}", min_value=0.0, value=1.0)

    forecast_features.append({
        "PRIX_DE_DETAIL": prix,
        "RABAIS": rabais,
        "NOMBRE_CLIENT_FIN_CONTRAT": clients_fin,
        "CROIS_BYOD": byod,
        "Fct_CROIS": facteur_crois
    })


if st.button("Lancer la prévision 🚀"):
    if historical_data is None:
        st.error("Veuillez téléverser un fichier CSV contenant les données historiques.")
    else:
        # Prepare payload
        payload = {
            "historical_data": historical_data.to_dict(orient="records"),
            "forecast_features": forecast_features
        }

        # Send request to API
        with st.spinner("Prévision en cours..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            predictions = response.json()["predictions"]
            st.success("Prévision terminée ✅")
            st.subheader("📊 Résultats des prévisions (ventes prévues par semaine) :")
            st.write(predictions)

            # Plot
            last_week_date = pd.to_datetime(historical_data["SEMAINE"].iloc[-1])
            future_weeks = [last_week_date + pd.Timedelta(weeks=i+1) for i in range(len(predictions))]
            plt.figure(figsize=(10, 5))
            plt.plot(future_weeks, predictions, marker='o')
            plt.xlabel("Semaine future")
            plt.ylabel("Ventes prédites")
            plt.title("Prévisions des ventes iPhone")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            plt.gcf().autofmt_xdate()
            st.pyplot(plt)

        else:
            st.error("Erreur lors de la prévision :")
            st.write(response.text)
