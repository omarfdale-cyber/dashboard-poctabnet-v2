import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

# ==========================
# 1. Configuration du dashboard
# ==========================
st.set_page_config(page_title="Dashboard POC – Prédiction des ventes Olist", layout="wide")
st.title("Preuve de Concept – Prédiction du montant total client (Olist)")
st.write("Ce dashboard compare RandomForest, XGBoost et TabNet pour la prédiction du montant total client.")

# ==========================
# 2. Chargement du dataset
# ==========================
@st.cache_data
def load_data():
    file_path = "clean_dataa.csv"  # ton fichier CSV (export du notebook)
    df = pd.read_csv(file_path)

    # Pour Streamlit Cloud : échantillonner 5000 lignes max
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    return df

df = load_data()
st.subheader("Aperçu du jeu de données")
st.dataframe(df.head())

# ==========================
# 3. Préparation des données
# ==========================
df['ratio_valeur'] = df['montant_total'] / (df['nb_produits_total'] + 1)
df['log_recence'] = np.log1p(df['recence'])
df['log_frequence'] = np.log1p(df['frequence'])
df['score_produit'] = df['review_score_moyen'] * df['nb_produits_total']
df = df[df['montant_total'] < df['montant_total'].quantile(0.99)]

X = df[['frequence', 'recence', 'nb_produits_total', 'review_score_moyen',
        'nb_reviews', 'ratio_valeur', 'log_recence', 'log_frequence', 'score_produit']]
y = df['montant_total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 4. Entraînement des modèles
# ==========================
st.subheader("Évaluation des modèles")
results = []

# ---- RandomForest ----
with st.spinner("Entraînement du modèle RandomForest..."):
    start = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rf_time = time.time() - start
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    results.append(["RandomForest", rf_r2, rf_rmse, rf_mae, rf_time])

# ---- XGBoost ----
with st.spinner("Entraînement du modèle XGBoost..."):
    start = time.time()
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    xgb_time = time.time() - start
    xgb_r2 = r2_score(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    results.append(["XGBoost", xgb_r2, xgb_rmse, xgb_mae, xgb_time])

# ---- TabNet ----
with st.spinner("Entraînement du modèle TabNet..."):
    X_train_tabnet, X_test_tabnet = X_train_scaled.copy(), X_test_scaled.copy()
    y_train_tabnet, y_test_tabnet = y_train.copy(), y_test.copy()

    model_tabnet = TabNetRegressor(seed=42, verbose=0)
    start = time.time()
    model_tabnet.fit(
        X_train_tabnet, y_train_tabnet.values.reshape(-1, 1),
        eval_set=[(X_test_tabnet, y_test_tabnet.values.reshape(-1, 1))],
        eval_metric=['rmse'],
        patience=10,
        max_epochs=30,
        batch_size=128,
        virtual_batch_size=64
    )
    y_pred_tabnet = model_tabnet.predict(X_test_tabnet).flatten()
    tabnet_time = time.time() - start
    tabnet_r2 = r2_score(y_test_tabnet, y_pred_tabnet)
    tabnet_rmse = np.sqrt(mean_squared_error(y_test_tabnet, y_pred_tabnet))
    tabnet_mae = mean_absolute_error(y_test_tabnet, y_pred_tabnet)
    results.append(["TabNet", tabnet_r2, tabnet_rmse, tabnet_mae, tabnet_time])

# ==========================
# 5. Résultats comparatifs
# ==========================
results_df = pd.DataFrame(results, columns=["Modèle", "R²", "RMSE", "MAE", "Temps (s)"])
st.dataframe(results_df.style.format({"R²": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "Temps (s)": "{:.2f}"}))

# Graphique R²
st.subheader("Comparaison des performances (R²)")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="Modèle", y="R²", data=results_df, palette="viridis", ax=ax)
ax.set_title("Performance en R²")
st.pyplot(fig)

# Graphique temps
st.subheader("Temps d'entraînement des modèles")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="Modèle", y="Temps (s)", data=results_df, palette="coolwarm", ax=ax)
ax.set_title("Durée d'exécution")
st.pyplot(fig)

# ==========================
# 6. Analyse des erreurs TabNet
# ==========================
st.subheader("Analyse du modèle TabNet")
residuals = y_test_tabnet - y_pred_tabnet

# Valeurs réelles vs prédites
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=y_test_tabnet, y=y_pred_tabnet, alpha=0.5, color='navy', ax=ax)
ax.plot([y_test_tabnet.min(), y_test_tabnet.max()],
        [y_test_tabnet.min(), y_test_tabnet.max()], 'r--')
ax.set_xlabel("Valeurs réelles")
ax.set_ylabel("Valeurs prédites")
ax.set_title("Valeurs réelles vs prédites (TabNet)")
st.pyplot(fig)

# Distribution des erreurs
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True, color='purple', ax=ax)
ax.set_title("Distribution des erreurs (résidus) - TabNet")
ax.set_xlabel("Erreur (réelle - prédite)")
st.pyplot(fig)

# ==========================
# 7. Conclusion
# ==========================
st.subheader("Conclusion")
st.write("""
Les trois modèles donnent d'excellents résultats :
- **RandomForest** : fiable et rapide  
- **XGBoost** : très bon compromis précision/rapidité  
- **TabNet** : plus lent mais plus interprétable et robuste sur les données tabulaires  

Ce POC prouve la faisabilité d’un modèle prédictif industrialisable pour estimer les montants d’achat client.
""")
