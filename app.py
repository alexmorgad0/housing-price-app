# app.py
import streamlit as st
import pandas as pd
from joblib import load
import json
from pathlib import Path

st.set_page_config(page_title="🏠 House Price (€/m²) Predictor", layout="centered")

MODEL_PATH = Path("rf_price_per_m2.joblib")
FEATURES_PATH = Path("rf_price_features.json")
CHOICES_PATH = Path("choices.json")
GDRIVE_FILE_ID = "1vPIL6uwvfknWttb4CzS8WDogmUI2i0nf"  # your Drive file id

def ensure_model():
    if MODEL_PATH.exists():
        return
    st.warning("Downloading model file (first run only)…")
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=str(MODEL_PATH), quiet=False)
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 10_000_000:
            raise RuntimeError("Downloaded file looks too small or missing.")
    except Exception as e:
        st.error(f"Model download failed: {e}")
        st.stop()

def read_json_resilient(path: Path, default):
    if not path.exists(): return default
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return default

ensure_model()

@st.cache_resource
def load_assets():
    features = read_json_resilient(FEATURES_PATH, None)
    if not features:
        st.error(f"Missing or invalid {FEATURES_PATH.name}."); st.stop()
    choices = read_json_resilient(CHOICES_PATH, {"Town": [], "Type": []})
    model = load(MODEL_PATH)

    # patch for sklearn 1.3 -> 1.7 (monotonic_cst)
    from sklearn.tree import DecisionTreeRegressor
    if hasattr(model, "steps"):
        last = model.steps[-1][1]
        if hasattr(last, "estimators_"):
            for est in last.estimators_:
                if isinstance(est, DecisionTreeRegressor) and not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
    return model, features, choices

model, features, choices = load_assets()

cat_cols = ["Town", "Type"]
num_cols = [c for c in features if c not in cat_cols]

st.title("🏠 House Price (€/m²) Predictor")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    inputs = {}

    # Town (searchable only)
    with c1:
        inputs["Town"] = st.selectbox(
            "Town (type to search)",
            options=choices.get("Town", []), index=None, placeholder="Start typing…"
        ) or ""

    # Type (searchable)
    with c2:
        inputs["Type"] = st.selectbox(
            "Type (type to search)",
            options=choices.get("Type", []), index=None, placeholder="Start typing…"
        ) or ""

    defaults = {
        "TotalArea": 80, "TotalRooms": 3, "NumberOfBathrooms": 1,
        "Parking": 0, "Elevator": 0,
        "drive_min_final": 20.0, "drive_km_final": 10.0,
        "no_transit_route": 0, "travel_min_final": 30.0,
    }

    # Left column
    with c1:
        inputs["TotalArea"] = st.number_input("Total Area", value=int(defaults["TotalArea"]),
                                              step=1, min_value=0, format="%d")
        inputs["TotalRooms"] = st.number_input("Number of Rooms", value=int(defaults["TotalRooms"]),
                                               step=1, min_value=0, format="%d")
        inputs["Parking"] = st.number_input("Number of Parking Spots", value=int(defaults["Parking"]),
                                            step=1, min_value=0, format="%d")
        inputs["drive_km_final"] = st.number_input("Distance by Car to City Center (km)",
                                                   value=float(defaults["drive_km_final"]))

    # Right column (set transport availability + car time first)
    with c2:
        inputs["NumberOfBathrooms"] = st.number_input("Number of Bathrooms", value=int(defaults["NumberOfBathrooms"]),
                                                      step=1, min_value=0, format="%d")
        inputs["Elevator"] = st.number_input("Has Elevator?  (0 = No, 1 = Yes)",
                                             value=int(defaults["Elevator"]), step=1, min_value=0, max_value=1, format="%d")
        inputs["drive_min_final"] = st.number_input("Travel Time To City Center by Car (minutes)",
                                                    value=float(defaults["drive_min_final"]))
        inputs["no_transit_route"] = st.number_input(
            "Has viable Transportation close by?  (1 = No transports, 0 = Has transports)",
            value=int(defaults["no_transit_route"]), step=1, min_value=0, max_value=1, format="%d"
        )

    # Transit time: show input only if transports exist; otherwise auto-use car time
    with c1:
        if inputs["no_transit_route"] == 1:
            inputs["travel_min_final"] = float(inputs["drive_min_final"])
            st.info("No viable transportation: using **car time** as transit time.")
        else:
            inputs["travel_min_final"] = st.number_input(
                "Travel Time To City Center by Transports (minutes)  — if none, we’ll use the car time",
                value=float(defaults["travel_min_final"])
            )

    submitted = st.form_submit_button("Predict")

st.caption("ℹ️ Travel times were computed assuming departure at **08:00** on a **weekday**.")

if submitted:
    # build row in training order
    row = {k: inputs.get(k, None) for k in features}
    X = pd.DataFrame([row])
    try:
        ppm2 = float(model.predict(X)[0])
        total = ppm2 * float(inputs.get("TotalArea", 0) or 0)
        st.success(f"Estimated price: €{total:,.0f}  (≈ €{ppm2:,.0f} / m²)")
        with st.expander("Show input row"):
            st.write(row)
    except Exception as e:
        st.error(f"Prediction failed: {e}")



