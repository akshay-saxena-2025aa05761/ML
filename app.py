import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.title("BITS Pilani - ML Assignment 2 Classifier")

# a. Dataset upload option [cite: 91]
uploaded_file = st.file_uploader("Upload Test CSV Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]

    # b. Model selection dropdown [cite: 92]
    model_choice = st.selectbox("Select Model", 
        ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"])

    if st.button("Run Prediction"):
        # Load the saved model
        with open(f'model/{model_choice}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_test)

        # c. Display evaluation metrics [cite: 93]
        st.subheader("Model Performance")
        st.write(f"Accuracy: {model.score(X_test, y_test):.4f}")

        # d. Confusion matrix and report [cite: 94]
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))