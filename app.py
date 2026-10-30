import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

# ---------- Page setup ----------
st.set_page_config(page_title="AFH Coupon Predictor", page_icon="☕", layout="wide")

st.title("Will Consumer use this Coupon?")
st.write(
    "Upload test data to see how five trained classification models predict whether "
    "a consumer will accept a coupon at coffee house, takeaway, restaurant or bar."
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
st.sidebar.header("Model Options")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()), index=4)
model = load_model(MODEL_FILES[model_name])

# ---------- Sidebar: project info and sample data ----------
st.sidebar.markdown("---")
st.sidebar.subheader("About this application")
st.sidebar.write(
    "Dataset: UCI In-Vehicle Coupon Recommendation. "
    "Target Y: 1 = coupon accepted, 0 = not accepted."
)
st.sidebar.write("Models were trained once in model_training.ipynb and are only loaded here (no retraining).")

# Let users download the test file so they can try the app straight away
try:
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test data", f,
                                   file_name="test_data.csv", mime="text/csv")
except FileNotFoundError:
    pass   # if the file is not in the repo, simply skip the button
# ---------- Upload test data ----------
uploaded = st.file_uploader("Upload test data (CSV)", type="csv")

if uploaded is None:
    st.info("Please upload a data (CSV) file to begin (for example, test_data.csv from the GitHub repo).")
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

# ---------- Predict using the saved model (no retraining) ----------
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]   # probability of "accepted", needed for AUC

# AUC needs both classes (0 and 1) in the uploaded data
auc = roc_auc_score(y, y_prob) if y.nunique() == 2 else None

# ---------- Show the 6 evaluation metrics ----------
st.subheader(f"Evaluation metrics – {model_name}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
c2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
c3.metric("Precision", f"{precision_score(y, y_pred, zero_division=0):.4f}")
c4.metric("Recall", f"{recall_score(y, y_pred, zero_division=0):.4f}")
c5.metric("F1 Score", f"{f1_score(y, y_pred, zero_division=0):.4f}")
c6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

# ---------- Confusion matrix and classification report side by side ----------
left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Orange",
                xticklabels=["Not accepted", "Accepted"],
                yticklabels=["Not accepted", "Accepted"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, labels=[0, 1],
                                   target_names=["Not accepted", "Accepted"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

# ---------- Predictions table with download ----------
st.subheader("Predictions")
predictions = data.copy()
predictions["Predicted"] = y_pred
predictions["Probability_accept"] = y_prob.round(3)
st.dataframe(predictions.head(20))
st.download_button("⬇️ Download all predictions", predictions.to_csv(index=False),
                   file_name="predictions.csv", mime="text/csv")

# ---------- Compare all five models on the uploaded data ----------
st.subheader("Compare all five models on this data")
if st.button("Run comparison"):
    rows = {}
    for name, path in MODEL_FILES.items():
        m = load_model(path)
        p = m.predict(X)
        prob = m.predict_proba(X)[:, 1]
        rows[name] = {
            "Accuracy": accuracy_score(y, p),
            "AUC": roc_auc_score(y, prob) if y.nunique() == 2 else float("nan"),
            "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0),
            "F1": f1_score(y, p, zero_division=0),
            "MCC": matthews_corrcoef(y, p),
        }
    comparison = pd.DataFrame(rows).T.round(4)
    st.dataframe(comparison)
    st.bar_chart(comparison[["Accuracy", "AUC", "F1", "MCC"]])
    best = comparison["MCC"].idxmax()
    st.success(f"Best model on this data (by MCC): {best}")
