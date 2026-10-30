import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="Coupon Acceptance Predictor", page_icon="☕", layout="wide")

st.title("☕ Away-From-Home Coupon Acceptance Predictor")
st.write(
    "Upload test data to see how five trained classification models predict whether "
    "a consumer will accept a food or drink coupon (coffee house, takeaway, restaurant or bar)."
)

# ---------- Saved models (trained in model_training.ipynb) ----------
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}
TARGET = "Y"   # 1 = coupon accepted, 0 = not accepted


@st.cache_resource   # load each model only once, not on every click
def load_model(path):
    return joblib.load(path)


# ---------- Sidebar: model selection ----------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()), index=4)
model = load_model(MODEL_FILES[model_name])

# ---------- Upload test data ----------
uploaded = st.file_uploader("Upload test data (CSV)", type="csv")

if uploaded is None:
    st.info("Please upload a CSV file to begin (for example, test_data.csv from the GitHub repo).")
    st.stop()   # nothing more to show until a file is uploaded

data = pd.read_csv(uploaded)
st.subheader("Uploaded data preview")
st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
st.dataframe(data.head())

# ---------- Check the file has the columns the model needs ----------
# The saved pipeline remembers which feature columns it was trained on
feature_cols = list(model.feature_names_in_)
missing = [col for col in feature_cols + [TARGET] if col not in data.columns]

if missing:
    st.error("The uploaded file is missing these required columns: " + ", ".join(missing))
    st.stop()

X = data[feature_cols]   # input features (extra columns are ignored)
y = data[TARGET]         # true answers, used to calculate metrics
st.success(f"All required columns found. Ready to evaluate the {model_name} model.")
