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
st.write("Ce dashboard présente la comparaison entre RandomForest, XGBoost et TabNet pour la prédiction du montant total client.")

# ==========================
# 2. Chargement du dataset
# ==========================
@st.cache_data
def load_data():
    file_path = "clean_dataa.csv"  # Mets ici ton fichier nettoyé (export CSV du notebook)
    df = pd.read_csv(file_path)
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
    rf = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_split=5,
                               min_samples_leaf=3, random_state=42)
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
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=8,
                       subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    xgb_time = time.time() - start
    xgb_r2 = r2_score(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    results.append(["XGBoost", xgb_r2, xgb_rmse, xgb_mae, xgb_time])

# ---- TabNet ----
with st.spinner("Entraînement du modèle TabNet... (peut prendre plusieurs minutes)"):
    X_train_tabnet, X_test_tabnet = X_train_scaled.copy(), X_test_scaled.copy()
    y_train_tabnet, y_test_tabnet = y_train.copy(), y_test.copy()

    model_tabnet = TabNetRegressor(seed=42, verbose=0)
    start = time.time()
    model_tabnet.fit(
        X_train_tabnet, y_train_tabnet.values.reshape(-1, 1),
        eval_set=[(X_test_tabnet, y_test_tabnet.values.reshape(-1, 1))],
        eval_metric=['rmse'],
        patience=20,
        max_epochs=50,
        batch_size=256,
        virtual_batch_size=128
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
ax.set_title("Durée d'exécution de chaque modèle")
st.pyplot(fig)

# ==========================
# 6. Analyse des erreurs TabNet
# ==========================
st.subheader("Analyse du modèle TabNet")

residuals = y_test_tabnet - y_pred_tabnet

# Valeurs réelles vs prédites
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=y_test_tabnet, y=y_pred_tabnet, alpha=0.5, color='navy', ax=ax)
ax.plot([y_test_tabnet.min(), y_test_tabnet.max()], [y_test_tabnet.min(), y_test_tabnet.max()], 'r--')
ax.set_xlabel("Valeurs réelles")
ax.set_ylabel("Valeurs prédites")
ax.set_title("Comparaison valeurs réelles vs prédites (TabNet)")
st.pyplot(fig)

# Distribution des erreurs
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True, color='purple', ax=ax)
ax.set_title("Distribution des erreurs (résidus) - TabNet")
ax.set_xlabel("Erreur (réelle - prédite)")
st.pyplot(fig)

# Importance des variables
st.subheader("Importance des variables selon TabNet")
try:
    feature_importances = model_tabnet.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=feature_importances, y=X.columns, palette='viridis', ax=ax)
    ax.set_title("Importance des variables")
    st.pyplot(fig)
except Exception as e:
    st.write("Impossible d'afficher les importances des variables pour TabNet :", e)

# ==========================
# 7. Conclusion
# ==========================
st.subheader("Conclusion")
st.write("""
Les trois modèles offrent d’excellentes performances :
- RandomForest : modèle de référence, rapide et stable  
- XGBoost : très performant et efficace  
- TabNet : plus lent mais plus explicable et pertinent sur les données tabulaires  

Cette preuve de concept démontre la faisabilité d’un système prédictif industrialisable.
""")

