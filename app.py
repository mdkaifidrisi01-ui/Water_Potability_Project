import pandas as pd
import numpy as np
import joblib
import streamlit as st

# 1. Page Configuration and Water-Themed CSS
st.set_page_config(page_title="Water Potability Prediction", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #F0F8FF; } /* Alice Blue */
    h1, h2, h3 { color: #005b96; } /* Deep Water Blue */
    div.stButton > button:first-child {
        background-color: #03396c;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #011f4b;
        color: white;
    }
    .result-card-safe {
        background-color: #d4edda; border-left: 5px solid #28a745; padding: 20px; border-radius: 5px; color: #155724;
    }
    .result-card-unsafe {
        background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 20px; border-radius: 5px; color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Safely Load the Trained SVM Pipeline
@st.cache_resource
def load_artifacts():
    # We only need to load ONE file now, because the SVM pipeline contains the scaler and imputer automatically!
    svm_pipeline = joblib.load('best_svm_pipeline.pkl')
    return svm_pipeline

try:
    model = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Error loading machine learning models: {e}. Please ensure the 'models' folder contains 'best_svm_pipeline.pkl'.")

# 3. Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict Potability", "Model Dashboard", "Project Info"])

# 4. Feature Input Constraints
feature_limits = {
    'ph': (0.0, 14.0, 7.0),
    'Hardness': (47.0, 324.0, 196.0),
    'Solids': (320.0, 61228.0, 20927.0),
    'Chloramines': (0.3, 13.2, 7.1),
    'Sulfate': (129.0, 482.0, 333.0),
    'Conductivity': (181.0, 754.0, 421.0),
    'Organic_carbon': (2.2, 28.3, 14.2),
    'Trihalomethanes': (0.7, 124.0, 66.6),
    'Turbidity': (1.4, 6.8, 3.9)
}

# ==========================================
# PAGE A: HOME
# ==========================================
if page == "Home":
    st.title("💧 Water Potability Prediction System")
    st.markdown("### Welcome to the Machine Learning Capstone Project!")
    st.write("""
    **Project Objective:** To develop a robust Machine Learning classification system capable of analyzing 9 chemical and physical water quality parameters to predict whether water is safe for human consumption.

    **What is Water Potability?**
    Potable water is water that is safe to drink or use for food preparation without risk of health problems. Determining potability traditionally requires expensive and time-consuming laboratory tests. This application demonstrates how Artificial Intelligence can assist in providing rapid, real-time safety estimates based on standard sensor data.
    """)
    st.info("👈 Please use the sidebar to navigate to the Prediction Tool or explore the Model Dashboard.")

# ==========================================
# PAGE B: WATER QUALITY PREDICTION
# ==========================================
elif page == "Predict Potability":
    st.title("🧪 Enter Water Quality Parameters")
    st.write("Adjust the sliders to input the chemical metrics of the water sample.")

    if not artifacts_loaded:
        st.warning("Cannot make predictions because the model files are missing.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.slider("pH Level", feature_limits['ph'][0], feature_limits['ph'][1], feature_limits['ph'][2])
            chloramines = st.slider("Chloramines (ppm)", feature_limits['Chloramines'][0], feature_limits['Chloramines'][1], feature_limits['Chloramines'][2])
            organic_carbon = st.slider("Organic Carbon (ppm)", feature_limits['Organic_carbon'][0], feature_limits['Organic_carbon'][1], feature_limits['Organic_carbon'][2])
        with col2:
            hardness = st.slider("Hardness (mg/L)", feature_limits['Hardness'][0], feature_limits['Hardness'][1], feature_limits['Hardness'][2])
            sulfate = st.slider("Sulfate (mg/L)", feature_limits['Sulfate'][0], feature_limits['Sulfate'][1], feature_limits['Sulfate'][2])
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", feature_limits['Trihalomethanes'][0], feature_limits['Trihalomethanes'][1], feature_limits['Trihalomethanes'][2])
        with col3:
            solids = st.slider("Solids / TDS (ppm)", feature_limits['Solids'][0], feature_limits['Solids'][1], feature_limits['Solids'][2])
            conductivity = st.slider("Conductivity (μS/cm)", feature_limits['Conductivity'][0], feature_limits['Conductivity'][1], feature_limits['Conductivity'][2])
            turbidity = st.slider("Turbidity (NTU)", feature_limits['Turbidity'][0], feature_limits['Turbidity'][1], feature_limits['Turbidity'][2])

        if st.button("Predict Water Safety", use_container_width=True):
            input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                                      columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

            # We pass the raw data straight to the model. The pipeline handles scaling and imputation automatically!
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-safe">
                    <h3>✅ SAFE TO DRINK (Potable)</h3>
                    <p>The Support Vector Machine predicts this water sample is safe for human consumption.</p>
                    <b>Confidence/Probability: {probability[1]*100:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-card-unsafe">
                    <h3>❌ NOT SAFE (Non-Potable)</h3>
                    <p>The Support Vector Machine predicts this water sample is highly toxic or unsuitable for consumption.</p>
                    <b>Confidence/Probability: {probability[0]*100:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)
            st.caption("Disclaimer: This tool is for academic demonstration purposes only and does not replace professional laboratory certification.")

# ==========================================
# PAGE C: MODEL PERFORMANCE DASHBOARD
# ==========================================
elif page == "Model Dashboard":
    st.title("📊 Model Performance")
    st.subheader("Final Model Metrics (Tuned Support Vector Machine)")

    # Using the exact metrics from your recent 6-model comparison output
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "61.28%")
    col2.metric("Precision", "50.37%")
    col3.metric("Recall", "52.73%")
    col4.metric("F1 Score", "0.5153")

    st.markdown("---")
    st.write("### Why SVM?")
    st.write("Unlike simple linear models, the Support Vector Machine (SVM) uses a Radial Basis Function (RBF) kernel to project the water data into higher dimensions. Because safe and unsafe water share overlapping chemical ranges, this algorithm was chosen for its mathematical ability to slice complex boundaries, achieving the highest test F1-Score of all models evaluated.")

# ==========================================
# PAGE D: PROJECT INFORMATION
# ==========================================
elif page == "Project Info":
    st.title("ℹ️ Project & Technology Stack")
    st.write("### Algorithms Evaluated (5-Fold Cross Validation)")
    st.write("1. **Logistic Regression:** Baseline linear model.")
    st.write("2. **XGBoost:** Gradient boosting algorithm.")
    st.write("3. **Random Forest:** Ensemble of decision trees.")
    st.write("4. **Decision Tree & KNN:** Evaluated as secondary models.")
    st.write("5. **Support Vector Machine (Selected):** Margin-based classifier. Selected due to its highest F1-score and robustness to overlapping data distributions.")

    st.write("### Technology Stack")
    st.write("- **Language:** Python 3")
    st.write("- **Data Processing:** Pandas, NumPy, imblearn (SMOTE)")
    st.write("- **Machine Learning:** Scikit-Learn (Pipelines, Imputers, SVM)")
    st.write("- **Web Application:** Streamlit")
