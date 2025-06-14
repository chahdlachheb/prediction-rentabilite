import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Chargement des données
df = pd.read_csv("dataset_projets_rentables.csv", encoding="latin")
df["rentable"] = df["rentable"].astype(int)

# Encoder la colonne 'secteur'
le = LabelEncoder()
df["secteur_encoded"] = le.fit_transform(df["secteur"])

# Préparation des données
X = df[["coÃ»t_projet", "revenu_estime", "duree_mois", "secteur_encoded", "nombre_employes"]]
y = df["rentable"]

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraînement
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=4)
model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarde
joblib.dump(model, "model_logistic.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Modèle et encodeurs sauvegardés.")
