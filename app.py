import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CancerSense AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 5px;
}
.hero-subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 8px 25px rgba(15,23,42,0.06);
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    color: #64748b;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_PATH = "models/cancer_model.pkl"

try:
    package = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file not found. Please run cancer.ipynb first so that "
        "models/cancer_model.pkl is created."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the model file: {e}")
    st.stop()

# The notebook saves ONE complete sklearn Pipeline.
# Therefore, do NOT try to load package["scaler"] separately.
model = package["model"]
model_name = package.get("model_name", "Machine Learning Model")
features = package["features"]
target_mapping = package.get("target_mapping", {"Benign": 0, "Malignant": 1})

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="hero-title">🎗️ CancerSense AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-subtitle">'
    'Machine Learning powered breast cancer classification'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "This application is an educational machine-learning demonstration "
    "and is not a medical diagnosis."
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient Information")

st.caption(
    "Enter the patient information using the same fields used to train "
    "the model."
)

# These are the actual input fields in cancer.csv.
values = {}

col1, col2, col3 = st.columns(3)

with col1:
    values["Year"] = st.selectbox("Year", ["2019", "2020"])
    values["Age"] = st.number_input(
        "Age", min_value=1, max_value=120, value=40, step=1
    )
    values["Menopause"] = st.selectbox(
        "Menopause", ["0", "1"],
        help="0 = No, 1 = Yes"
    )

with col2:
    values["Tumor Size (cm)"] = st.number_input(
        "Tumor Size (cm)", min_value=0.0, max_value=100.0,
        value=3.0, step=0.5
    )
    values["Inv-Nodes"] = st.number_input(
        "Inv-Nodes", min_value=0, max_value=100, value=0, step=1
    )
    values["Breast"] = st.selectbox("Breast", ["Right", "Left"])

with col3:
    values["Metastasis"] = st.selectbox(
        "Metastasis", ["0", "1"],
        help="0 = No, 1 = Yes"
    )
    values["Breast Quadrant"] = st.selectbox(
        "Breast Quadrant",
        ["Upper inner", "Upper outer", "Lower outer", "Lower inner"]
    )
    values["History"] = st.selectbox(
        "History", ["0", "1"],
        help="0 = No, 1 = Yes"
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔍 ANALYZE CANCER RISK", use_container_width=True):

    # IMPORTANT:
    # Pass the ORIGINAL, unscaled values to the pipeline.
    # The pipeline automatically performs missing-value handling,
    # encoding and scaling exactly as it did during training.
    input_data = pd.DataFrame([values], columns=features)

    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            # The pipeline was trained with 0 = Benign and 1 = Malignant.
            malignant_index = classes.index(target_mapping["Malignant"])
            probability = float(probabilities[malignant_index])
        else:
            probability = None

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if str(prediction) == str(target_mapping["Malignant"]):
            st.error("⚠️ Malignant Pattern Predicted")
        else:
            st.success("✅ Benign Pattern Predicted")

    with col2:
        if probability is not None:
            st.metric(
                "Estimated Malignant Probability",
                f"{probability * 100:.1f}%"
            )

    # --------------------------------------------------
    # GAUGE
    # --------------------------------------------------
    if probability is not None:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Malignant Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {},
                    "steps": [
                        {"range": [0, 40]},
                        {"range": [40, 70]},
                        {"range": [70, 100]}
                    ]
                }
            )
        )

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------
st.markdown("---")
st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", model_name)
col2.metric("Features", len(features))
col3.metric("Learning Type", "Supervised ML")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="footer">'
    'CancerSense AI • Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
