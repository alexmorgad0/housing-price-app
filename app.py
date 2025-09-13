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

# ---------- ensure model present ----------
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

# ---------- robust JSON loader ----------
def read_json_resilient(path: Path, default):
    if not path.exists():
        return default
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return json.load(f)
        except Exception:
            continue
    return default

# ---------- sklearn 1.3 -> 1.7 compatibility shim ----------
def patch_forest_monotonic(model):
    """Add missing .monotonic_cst attribute to DecisionTreeRegressor estimators."""
    try:
        from sklearn.tree import DecisionTreeRegressor
        # RandomForestRegressor
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if isinstance(est, DecisionTreeRegressor) and not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
        # Pipeline -> last step could be a RF
        if hasattr(model, "steps"):
            last = model.steps[-1][1]
            if hasattr(last, "estimators_"):
                for est in last.estimators_:
                    if isinstance(est, DecisionTreeRegressor) and not hasattr(est, "monotonic_cst"):
                        est.monotonic_cst = None
    except Exception:
        pass
    return model

ensure_model()

@st.cache_resource
def load_assets():
    features = read_json_resilient(FEATURES_PATH, None)
    if not features:
        st.error(f"Missing or invalid {FEATURES_PATH.name}.")
        st.stop()
    choices = read_json_resilient(CHOICES_PATH, {"Town": [], "Type": []})
    model = load(MODEL_PATH)
    model = patch_forest_monotonic(model)
    return model, features, choices

model, features, choices = load_assets()

# -------------------- UI --------------------
cat_cols = ["Town", "Type"]
num_cols = [c for c in features if c not in cat_cols]

st.title("🏠 House Price (€/m²) Predictor")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    inputs = {}

    with c1:
        town_opts = choices.get("Town", []) or []
        town_sel = st.selectbox(
            "Town (type to search)" if town_opts else "Town (no list available, type below)",
            options=town_opts, index=None, placeholder="Start typing…" if town_opts else None
        )
        town_manual = st.text_input("Or enter Town manually", value="")
        inputs["Town"] = (town_manual.strip() or (town_sel or "")).strip()

    with c2:
        type_opts = choices.get("Type", []) or []
        type_sel = st.selectbox(
            "Type (type to search)" if type_opts else "Type (no list available, type below)",
            options=type_opts, index=None, placeholder="Start typing…" if type_opts else None
        )
        type_manual = "" if type_opts else st.text_input("Or enter Type manually", value="")
        inputs["Type"] = (type_manual.strip() or (type_sel or "")).strip()

    defaults = {
        "TotalArea": 80, "TotalRooms": 3, "NumberOfBathrooms": 1, "Parking": 0, "Elevator": 0,
        "travel_min_final": 30.0, "drive_min_final": 20.0, "drive_km_final": 10.0, "no_transit_route": 0
    }

    with c1:
        inputs["TotalArea"] = st.number_input("Total Area", value=int(defaults["TotalArea"]), step=1, min_value=0, format="%d")
        inputs["TotalRooms"] = st.number_input("Number of Rooms", value=int(defaults["TotalRooms"]), step=1, min_value=0, format="%d")
        inputs["Parking"] = st.number_input("Number of Parking Spots", value=int(defaults["Parking"]), step=1, min_value=0, format="%d")
        inputs["travel_min_final"] = st.number_input("Travel Time To City Center by Transports (minutes)", value=float(defaults["travel_min_final"]))
        inputs["drive_km_final"] = st.number_input("Distance by Car to City Center (km)", value=float(defaults["drive_km_final"]))

    with c2:
        inputs["NumberOfBathrooms"] = st.number_input("Number of Bathrooms", value=int(defaults["NumberOfBathrooms"]), step=1, min_value=0, format="%d")
        inputs["Elevator"] = st.number_input("Has Elevator?  (0 = No, 1 = Yes)", value=int(defaults["Elevator"]), step=1, min_value=0, max_value=1, format="%d")
        inputs["drive_min_final"] = st.number_input("Travel Time To City Center by Car (minutes)", value=float(defaults["drive_min_final"]))
        inputs["no_transit_route"] = st.number_input("Has viable Transportation close by?  (1 = No transports, 0 = Has transports)", value=int(defaults["no_transit_route"]), step=1, min_value=0, max_value=1, format="%d")

    submitted = st.form_submit_button("Predict")

st.caption("ℹ️ Travel times were computed assuming departure at **08:00** on a **weekday**.")

if submitted:
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



