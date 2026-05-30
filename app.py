import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0d1f3c 0%, #0a0e1a 60%);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.hero-tag {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.4);
    color: #60a5fa;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 2rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

/* Section cards */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Streamlit widget overrides */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stDateInput"] > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: rgba(96,165,250,0.5) !important;
}
label { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 400 !important; }

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.4) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 6px 32px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Result panels */
.result-fraud {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(153,27,27,0.1));
    border: 1px solid rgba(220,38,38,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-clean {
    background: linear-gradient(135deg, rgba(22,163,74,0.15), rgba(21,128,61,0.1));
    border: 1px solid rgba(22,163,74,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.result-fraud .result-label { color: #f87171; }
.result-clean .result-label { color: #4ade80; }
.result-sub { color: #94a3b8; font-size: 0.9rem; }

/* Gauge */
.gauge-wrap { margin: 1.5rem 0 0.5rem; }
.gauge-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.gauge-track {
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.gauge-fill-fraud {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    transition: width 1s ease;
}
.gauge-fill-clean {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #34d399, #22c55e);
    transition: width 1s ease;
}
.gauge-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    margin-top: 0.75rem;
}
.result-fraud .gauge-pct { color: #f87171; }
.result-clean .gauge-pct { color: #4ade80; }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(96,165,250,0.3), transparent);
    margin: 2rem 0;
}

/* Info pills */
.pill {
    display: inline-block;
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 99px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    color: #93c5fd;
    margin: 0.25rem 0.25rem 0 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    full_pipeline = joblib.load('full_pipeline.pkl')
    imputer       = joblib.load('num_imputer.pkl')
    preprocessor  = joblib.load('preprocessor_pipeline.pkl')
    return full_pipeline, imputer, preprocessor

try:
    full_pipeline, imputer, preprocessor = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ── Feature engineering ─────────────────────────────────────────────────────────
def eng_features(df):
    df = df.copy()
    date_cols = []
    if 'policy_bind_date' in df.columns:
        date_cols.append('policy_bind_date')
    if 'incident_date' in df.columns:
        date_cols.append('incident_date')
    for d in date_cols:
        df[d] = pd.to_datetime(df[d], errors='coerce')
    if 'policy_bind_date' in df.columns and 'incident_date' in df.columns:
        df['policy_bind_year']  = df['policy_bind_date'].dt.year
        df['policy_bind_month'] = df['policy_bind_date'].dt.month
        df['incident_year']     = df['incident_date'].dt.year
        df['incident_month']    = df['incident_date'].dt.month
        df['days_between_incident_and_policy'] = (
            df['incident_date'] - df['policy_bind_date']
        ).dt.days
    if 'incident_hour_of_the_day' in df.columns:
        df['incident_hour_of_the_day'] = pd.to_numeric(
            df['incident_hour_of_the_day'], errors='coerce')
    return df


# ── Predict ─────────────────────────────────────────────────────────────────────
def predict_insurance(input_df):
    df = eng_features(input_df)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[num_cols] = imputer.transform(df[num_cols])
    X_processed  = preprocessor.transform(df)
    prediction   = full_pipeline.predict(X_processed)
    probability  = full_pipeline.predict_proba(X_processed)
    return int(prediction[0]), float(probability[0][1])


# ── UI ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🛡️ AI-Powered Detection</div>
    <h1>Insurance Fraud Detector</h1>
    <p>Fill in the claim details below and let the model assess fraud risk instantly.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model artifacts. Make sure the .pkl files are in the same folder as app.py.\n\nError: {load_error}")
    st.stop()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Input form ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:

    # Customer Info
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Customer Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=18, max_value=100, value=35)
    insured_sex = c2.selectbox("Gender", ["MALE", "FEMALE"])
    insured_education_level = c3.selectbox("Education Level", [
        "High School", "Associate", "College", "Bachelor", "Masters", "MD", "PhD"])
    c4, c5 = st.columns(2)
    insured_occupation = c4.selectbox("Occupation", [
        "craft-repair", "machine-op-inspct", "sales", "armed-forces",
        "tech-support", "prof-specialty", "other-service", "adm-clerical",
        "exec-managerial", "priv-house-serv", "protective-serv",
        "transport-moving", "handlers-cleaners", "farming-fishing"])
    insured_relationship = c5.selectbox("Relationship", [
        "husband", "wife", "own-child", "not-in-family", "other-relative", "unmarried"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Policy Info
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Policy Info</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    months_as_customer   = p1.number_input("Months as Customer", min_value=0, max_value=600, value=24)
    policy_deductable    = p2.selectbox("Deductible ($)", [500, 1000, 2000])
    policy_annual_premium = p3.number_input("Annual Premium ($)", min_value=0.0, value=1200.0, step=50.0)
    p4, p5 = st.columns(2)
    policy_state  = p4.selectbox("Policy State", ["OH", "IL", "IN", "VA", "NY", "SC", "WV", "NC", "PA"])
    umbrella_limit = p5.number_input("Umbrella Limit ($)", min_value=-1000000, max_value=10000000, value=0, step=100000)
    pb1, pb2 = st.columns(2)
    policy_bind_date = pb1.date_input("Policy Bind Date", value=date(2015, 1, 1), min_value=date(1990, 1, 1), max_value=date.today())
    st.markdown('</div>', unsafe_allow_html=True)

    # Incident Info
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚨 Incident Info</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    incident_date = i1.date_input("Incident Date", value=date(2015, 3, 15))
    incident_hour_of_the_day = i2.number_input("Incident Hour (0-23)", min_value=0, max_value=23, value=12)
    i3, i4 = st.columns(2)
    incident_type = i3.selectbox("Incident Type", [
        "Single Vehicle Collision", "Vehicle Theft",
        "Multi-vehicle Collision", "Parked Car"])
    incident_severity = i4.selectbox("Incident Severity", [
        "Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
    i5, i6, i7 = st.columns(3)
    collision_type = i5.selectbox("Collision Type", ["Front Collision", "Rear Collision", "Side Collision", "?"])
    authorities_contacted = i6.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"])
    number_of_vehicles_involved = i7.number_input("Vehicles Involved", min_value=1, max_value=10, value=1)
    i8, i9, i10 = st.columns(3)
    property_damage     = i8.selectbox("Property Damage", ["YES", "NO", "?"])
    bodily_injuries     = i9.number_input("Bodily Injuries", min_value=0, max_value=5, value=0)
    witnesses           = i10.number_input("Witnesses", min_value=0, max_value=10, value=0)
    police_report_available = st.selectbox("Police Report Available", ["YES", "NO", "?"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Claim Info
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Claim Info</div>', unsafe_allow_html=True)
    cl1, cl2 = st.columns(2)
    total_claim_amount = cl1.number_input("Total Claim ($)", min_value=0.0, value=50000.0, step=1000.0)
    injury_claim       = cl2.number_input("Injury Claim ($)", min_value=0.0, value=10000.0, step=500.0)
    cl3, cl4 = st.columns(2)
    property_claim = cl3.number_input("Property Claim ($)", min_value=0.0, value=15000.0, step=500.0)
    vehicle_claim  = cl4.number_input("Vehicle Claim ($)", min_value=0.0, value=25000.0, step=500.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Vehicle Info
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚗 Vehicle Info</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    auto_make = v1.selectbox("Auto Make", [
        "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Audi",
        "Nissan", "Dodge", "Suburu", "Jeep", "Mercedes", "Saab",
        "Volkswagen", "Accura"])
    auto_year = v2.number_input("Auto Year", min_value=1990, max_value=2025, value=2015)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Result column ─────────────────────────────────────────────────────────────
with right_col:
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    st.markdown("### Run Prediction")
    st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-bottom:1.5rem;'>Complete the form and click below to analyse this claim for fraud risk.</p>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Analyse Claim", use_container_width=True)

    if predict_btn:
        # Build input dataframe with ALL columns the model expects
        # Columns not shown in UI are filled with sensible defaults
        input_data = {
            'months_as_customer': [months_as_customer],
            'age': [age],
            'policy_bind_date': [str(policy_bind_date)],
            'policy_state': [policy_state],
            'policy_csl': ['250/500'],
            'policy_deductable': [policy_deductable],
            'policy_annual_premium': [policy_annual_premium],
            'umbrella_limit': [umbrella_limit],
            'insured_zip': [432700],
            'insured_sex': [insured_sex],
            'insured_education_level': [insured_education_level],
            'insured_occupation': [insured_occupation],
            'insured_hobbies': ['reading'],
            'insured_relationship': [insured_relationship],
            'capital-gains': [0],
            'capital-loss': [0],
            'incident_date': [str(incident_date)],
            'incident_type': [incident_type],
            'collision_type': [collision_type],
            'incident_severity': [incident_severity],
            'authorities_contacted': [authorities_contacted],
            'incident_state': ['OH'],
            'incident_city': ['Columbus'],
            'incident_hour_of_the_day': [incident_hour_of_the_day],
            'number_of_vehicles_involved': [number_of_vehicles_involved],
            'property_damage': [property_damage],
            'bodily_injuries': [bodily_injuries],
            'witnesses': [witnesses],
            'police_report_available': [police_report_available],
            'total_claim_amount': [total_claim_amount],
            'injury_claim': [injury_claim],
            'property_claim': [property_claim],
            'vehicle_claim': [vehicle_claim],
            'auto_make': [auto_make],
            'auto_model': ['A3'],
            'auto_year': [auto_year],
        }

        input_df = pd.DataFrame(input_data)

        with st.spinner("Analysing claim..."):
            try:
                prediction, fraud_prob = predict_insurance(input_df)
                no_fraud_prob = 1 - fraud_prob

                if prediction == 1:
                    pct = int(fraud_prob * 100)
                    st.markdown(f"""
                    <div class="result-fraud">
                        <div class="result-icon">🚨</div>
                        <div class="result-label">Fraud Detected</div>
                        <div class="result-sub">This claim shows signs of fraudulent activity</div>
                        <div class="gauge-wrap">
                            <div class="gauge-label"><span>Risk Level</span><span>{pct}%</span></div>
                            <div class="gauge-track">
                                <div class="gauge-fill-fraud" style="width:{pct}%"></div>
                            </div>
                        </div>
                        <div class="gauge-pct">{pct}%</div>
                        <div class="result-sub">Fraud Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    pct = int(no_fraud_prob * 100)
                    st.markdown(f"""
                    <div class="result-clean">
                        <div class="result-icon">✅</div>
                        <div class="result-label">No Fraud Detected</div>
                        <div class="result-sub">This claim appears to be legitimate</div>
                        <div class="gauge-wrap">
                            <div class="gauge-label"><span>Confidence</span><span>{pct}%</span></div>
                            <div class="gauge-track">
                                <div class="gauge-fill-clean" style="width:{pct}%"></div>
                            </div>
                        </div>
                        <div class="gauge-pct">{pct}%</div>
                        <div class="result-sub">Legitimacy Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Summary pills
                st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
                st.markdown(f"""
                <span class="pill">Age: {age}</span>
                <span class="pill">Claim: ${total_claim_amount:,.0f}</span>
                <span class="pill">{incident_type}</span>
                <span class="pill">{incident_severity}</span>
                <span class="pill">Witnesses: {witnesses}</span>
                <span class="pill">Police Report: {police_report_available}</span>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            margin-top: 1rem;
            color: #475569;
        ">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🛡️</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:#64748b;">
                Awaiting Claim Data
            </div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">
                Fill in the form and click<br>Analyse Claim to get a prediction
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model info card
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.8;
    ">
        <span style="color:#60a5fa; font-family:'Syne',sans-serif; font-weight:700; font-size:0.7rem; 
                     letter-spacing:0.1em; text-transform:uppercase;">Model Info</span><br>
        Algorithm: SelectFromModel + Random Forest<br>
        Tuned Accuracy: 85% &nbsp;|&nbsp; F1 Score: 0.85<br>
        Training: SMOTE-balanced dataset
    </div>
    """, unsafe_allow_html=True)
