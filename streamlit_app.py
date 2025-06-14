import streamlit as st
import numpy as np
import joblib

# Charger le modèle et les objets
model = joblib.load("model_logistic.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("📊 Prédiction de la Rentabilité d’un Projet")

# Formulaire
cout = st.number_input("💰 Coût du projet", 1000, 1_000_000, 150000)
revenu = st.number_input("📈 Revenu estimé", 1000, 2_000_000, 200000)
duree = st.slider("⏳ Durée (mois)", 1, 60, 12)
secteur = st.selectbox("🏢 Secteur", label_encoder.classes_)
employes = st.slider("👥 Nombre d'employés", 1, 500, 10)

if st.button("🔍 Prédire"):
    secteur_enc = label_encoder.transform([secteur])[0]
    features = np.array([[cout, revenu, duree, secteur_enc, employes]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]

    if pred == 1:
        st.success(f"✅ Le projet est **rentable** (probabilité : {proba:.2%})")
    else:
        st.error(f"❌ Le projet est **non rentable** (probabilité : {proba:.2%})")

    st.write("### Détails")
    st.write(f"**Probabilité rentable :** {proba:.2%}")
    st.write(f"**Probabilité non rentable :** {1 - proba:.2%}")
