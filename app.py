import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "telco_churn_pipeline.joblib"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"`{MODEL_PATH}` not found. Run the notebook's save cell and put the file "
        "in the same folder as app.py."
    )
    st.stop()

YES_IDX = list(model.classes_).index("Yes")

st.title("📉 Telco Customer Churn Predictor")
st.caption("Tuned Random Forest pipeline trained on the IBM Telco dataset.")

tab_single, tab_batch = st.tabs(["Single customer", "Batch (CSV)"])


def risk_label(p):
    if p >= 0.6:
        return "High", "🔴"
    if p >= 0.35:
        return "Medium", "🟠"
    return "Low", "🟢"


# ---------------- Single prediction ----------------
with tab_single:
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.subheader("Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet service", ["Fiber optic", "DSL", "No"]
            )
            addon_opts = ["No", "Yes", "No internet service"]
            security = st.selectbox("Online security", addon_opts)
            backup = st.selectbox("Online backup", addon_opts)
            protection = st.selectbox("Device protection", addon_opts)
            support = st.selectbox("Tech support", addon_opts)
            tv = st.selectbox("Streaming TV", addon_opts)
            movies = st.selectbox("Streaming movies", addon_opts)

        with c3:
            st.subheader("Account")
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5
            )
            total = st.number_input(
                "Total charges ($)",
                0.0,
                10000.0,
                float(round(monthly * max(tenure, 1), 2)),
                step=1.0,
            )

        submitted = st.form_submit_button("Predict churn", use_container_width=True)

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiple,
                    "InternetService": internet,
                    "OnlineSecurity": security,
                    "OnlineBackup": backup,
                    "DeviceProtection": protection,
                    "TechSupport": support,
                    "StreamingTV": tv,
                    "StreamingMovies": movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                }
            ]
        )

        prob = float(model.predict_proba(row)[0][YES_IDX])
        pred = model.predict(row)[0]
        level, icon = risk_label(prob)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Prediction", "Will churn" if pred == "Yes" else "Will stay")
        m3.metric("Risk level", f"{icon} {level}")
        st.progress(min(max(prob, 0.0), 1.0))

        if pred == "Yes":
            st.warning(
                "This customer is likely to leave. Consider a retention offer, "
                "a longer-term contract discount, or a support check-in."
            )
        else:
            st.success("This customer is likely to stay.")

# ---------------- Batch prediction ----------------
with tab_batch:
    st.write(
        "Upload a CSV with the same columns as the original Telco dataset "
        "(`customerID` and `Churn` are optional and ignored for prediction)."
    )
    file = st.file_uploader("CSV file", type="csv")

    if file is not None:
        df = pd.read_csv(file)
        X = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()

        # Same cleaning step as the notebook
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(
                X["TotalCharges"].astype(str).str.strip(), errors="coerce"
            )
            X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())

        try:
            probs = model.predict_proba(X)[:, YES_IDX]
        except Exception as e:
            st.error(f"Could not score this file: {e}")
        else:
            out = df.copy()
            out["Churn_Probability"] = probs.round(4)
            out["Prediction"] = model.predict(X)
            out["Risk"] = [risk_label(p)[0] for p in probs]

            k1, k2, k3 = st.columns(3)
            k1.metric("Customers", len(out))
            k2.metric("Predicted to churn", int((out["Prediction"] == "Yes").sum()))
            k3.metric("Avg churn probability", f"{probs.mean():.1%}")

            st.dataframe(
                out.sort_values("Churn_Probability", ascending=False),
                use_container_width=True,
            )
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv",
                "text/csv",
            )