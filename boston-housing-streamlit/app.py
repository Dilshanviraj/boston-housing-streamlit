import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & data
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("data/boston.csv")

st.set_page_config(page_title="Boston Housing Price Prediction", layout="wide")

# Sidebar Navigation
menu = ["Home", "Data Exploration", "Visualisations", "Predict", "Model Performance"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "Home":
    st.title("🏠 Boston Housing Price Prediction")
    st.write("This app predicts **Boston house prices** based on various features using a trained Random Forest model.")

elif choice == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write(df.describe())

    if st.checkbox("Show correlation heatmap"):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

elif choice == "Visualisations":
    st.subheader("Visualisations")
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    sns.histplot(df['PRICE'], bins=20, kde=True, ax=ax[0])
    ax[0].set_title("Distribution of Prices")
    sns.scatterplot(x=df['RM'], y=df['PRICE'], ax=ax[1])
    ax[1].set_title("Rooms vs Price")
    st.pyplot(fig)

elif choice == "Predict":
    st.subheader("Make a Prediction")
    features = []
    for col in df.columns[:-1]:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        features.append(val)

    if st.button("Predict Price"):
        prediction = model.predict([features])[0]
        st.success(f"Estimated Price: ${prediction*1000:,.2f}")

elif choice == "Model Performance":
    st.subheader("Model Performance Metrics")
    st.write("Random Forest was chosen based on cross-validation performance.")
    st.code("""
Random Forest R²: ~0.85
Random Forest MSE: ~10
    """)
    st.write("### Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": df.columns[:-1], "Importance": importances})
    st.bar_chart(feat_df.set_index("Feature"))