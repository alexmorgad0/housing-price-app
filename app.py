import streamlit as st
import pandas as pd
from joblib import load
import json
from pathlib import Path
import requests

st.set_page_config(page_title="🏠 House Price (€/m²) Predictor", layout="centered")

MODEL_PATH = Path("rf_price_per_m2.joblib")
if not MODEL_PATH.exists():
    st.warning("Downloading model file... (first run only)")
    url = "https://drive.google.com/file/d/1vPIL6uwvfknWttb4CzS8WDogmUI2i0nf/view?usp=drive_link"
    r = requests.get(url)
    MODEL_PATH.write_bytes(r.content)
FEATURES_PATH = "rf_price_features.json"
CHOICES_PATH = "choices.json"

@st.cache_resource
def load_assets():
    model = load(MODEL_PATH)
    features = json.load(open(FEATURES_PATH))
    choices = json.load(open(CHOICES_PATH)) if Path(CHOICES_PATH).exists() else {}
    return model, features, choices

model, features, choices = load_assets()

# Original feature names in your model
cat_cols = ["Town", "Type"]
num_cols = [c for c in features if c not in cat_cols]

st.title("🏠 House Price (€/m²) Predictor")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    inputs = {}

    # --- Town (searchable) + manual override ---
    with c1:
        town_opts = choices.get("Town", [])
        town_sel = st.selectbox(
            "Town (type to search)",
            options=town_opts,
            index=None,
            placeholder="Start typing…"
        )
        town_manual = st.text_input("Or enter Town manually (optional)", value="")
        inputs["Town"] = (town_manual.strip() or (town_sel or "")).strip()

    # --- Type (searchable) ---
    with c2:
        type_opts = choices.get("Type", [])
        type_sel = st.selectbox(
            "Type (type to search)",
            options=type_opts,
            index=None,
            placeholder="Start typing…"
        )
        inputs["Type"] = (type_sel or "").strip()

    # --- Numeric fields with requested labels/behavior ---
    # defaults
    defaults = {
        "TotalArea": 80,                # int only
        "TotalRooms": 3,
        "NumberOfBathrooms": 1,
        "Parking": 0,
        "Elevator": 0,
        "travel_min_final": 30.0,
        "drive_min_final": 20.0,
        "drive_km_final": 10.0,
        "no_transit_route": 0
    }

    # Left column fields
    with c1:
        inputs["TotalArea"] = st.number_input("Total Area", value=int(defaults["TotalArea"]),
                                              step=1, min_value=0, format="%d")
        inputs["TotalRooms"] = st.number_input("Number of Rooms", value=int(defaults["TotalRooms"]),
                                               step=1, min_value=0, format="%d")
        inputs["Parking"] = st.number_input("Number of Parking Spots", value=int(defaults["Parking"]),
                                            step=1, min_value=0, format="%d")
        inputs["travel_min_final"] = st.number_input(
            "Travel Time To City Center by Transports (minutes)",
            value=float(defaults["travel_min_final"])
        )
        inputs["drive_km_final"] = st.number_input("Distance by Car to City Center (km)",
                                                   value=float(defaults["drive_km_final"]))

    # Right column fields
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

    submitted = st.form_submit_button("Predict")

st.caption("ℹ️ Travel times were computed assuming departure at **08:00** on a **weekday**.")

if submitted:
    # Build the row strictly in training feature order
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

