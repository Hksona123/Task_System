# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
main_model = joblib.load("model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Optional: Load additional models
models = {}
try:
    models['rf_model'] = joblib.load("final_task_classifier.pkl")
    models['xgb_model'] = joblib.load("final_priority_predictor.pkl")
    models['scaler'] = joblib.load("scaler.pkl")
    additional_models_loaded = True
except:
    additional_models_loaded = False

# UI Settings
st.set_page_config(page_title="AI Task Management Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .css-18e3th9 {
            background-color: #1e1e1e;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Task Management System")

# Sidebar
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Process and Display
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    if "Task" in df.columns:
        st.success("'Task' column found. Running prediction...")

        X_transformed = tfidf_vectorizer.transform(df['Task'])
        df['Predicted_Label'] = main_model.predict(X_transformed)

        st.subheader("🧠 Task Label Prediction")
        st.dataframe(df[['Task', 'Predicted_Label']])

        st.subheader("📊 Label Distribution")
        st.bar_chart(df['Predicted_Label'].value_counts())

        # Task Summary Grouped
        st.subheader("📋 Task Summary by Label")
        st.dataframe(df.groupby('Predicted_Label')['Task'].count().reset_index(name='Task Count'))

        # Correlation if numerical columns exist
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            st.subheader("🔗 Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(num_df.corr(), annot=True, cmap='viridis', ax=ax)
            st.pyplot(fig)

        # Additional Models
        if additional_models_loaded and all(col in df.columns for col in ['Age', 'Income', 'Recency', 'Kidhome', 'Teenhome', 'Family_Size']):
            st.subheader("🔬 Advanced Predictions")
            features = ['Age', 'Income', 'Recency', 'Kidhome', 'Teenhome', 'Family_Size']
            scaled_features = models['scaler'].transform(df[features])

            df['Predicted_Category'] = models['rf_model'].predict(scaled_features)
            df['Predicted_Priority'] = models['xgb_model'].predict(scaled_features)

            st.dataframe(df[['Task', 'Predicted_Label', 'Predicted_Category', 'Predicted_Priority']])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.countplot(x='Predicted_Category', data=df, ax=axes[0])
            axes[0].set_title("Task Category Count")

            sns.countplot(x='Predicted_Priority', data=df, ax=axes[1])
            axes[1].set_title("Task Priority Count")
            st.pyplot(fig)

            # Summary Grouped
            st.subheader("📋 Task Summary by Priority")
            st.dataframe(df.groupby('Predicted_Priority')['Task'].count().reset_index(name='Task Count'))

            # Accuracy (Optional: if you want to display static scores)
            st.subheader("📈 Model Accuracy")
            st.write("Random Forest Classifier (Task Category): ~0.35")
            st.write("XGBoost Classifier (Priority): ~0.40")

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, file_name="task_output.csv", mime="text/csv")

    else:
        st.error("❌ 'Task' column not found in uploaded file. Include a 'Task' column.")
else:
    st.info("📥 Upload a CSV to get started.")