import streamlit as st

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Urban Safety Prediction System",
    page_icon="🏙️",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================

st.title("🏙️ Integrated Urban Safety Prediction System")

st.write(
    "Predict urban safety risk using air quality, traffic "
    "and accident severity."
)

st.divider()

# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader("Enter Urban Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    air_quality = st.selectbox(
        "Air Quality",
        ["Good", "Moderate", "Poor", "Hazardous"]
    )

with col2:
    traffic = st.selectbox(
        "Traffic Congestion",
        ["Low", "Medium", "High"]
    )

with col3:
    accident_severity = st.selectbox(
        "Accident Severity",
        [1, 2, 3, 4]
    )

# ==========================================================
# SCORE CALCULATION
# ==========================================================

air_scores = {
    "Good": 0,
    "Moderate": 33.33,
    "Poor": 66.67,
    "Hazardous": 100
}

traffic_scores = {
    "Low": 0,
    "Medium": 50,
    "High": 100
}

air_score = air_scores[air_quality]

traffic_score = traffic_scores[traffic]

accident_score = ((accident_severity - 1) / 3) * 100

overall_score = (
    air_score * 0.35
    + traffic_score * 0.30
    + accident_score * 0.35
)

# ==========================================================
# RISK LEVEL
# ==========================================================

if overall_score <= 33:
    risk_level = "Low Risk"
elif overall_score <= 66:
    risk_level = "Moderate Risk"
else:
    risk_level = "High Risk"

# ==========================================================
# RESULTS
# ==========================================================

st.divider()

st.subheader("Urban Safety Result")

result1, result2 = st.columns(2)

with result1:
    st.metric(
        "Overall Urban Risk Score",
        f"{overall_score:.2f}/100"
    )

with result2:
    st.metric(
        "Risk Level",
        risk_level
    )

# ==========================================================
# INPUT SUMMARY
# ==========================================================

st.subheader("Prediction Summary")

st.write("**Air Quality:**", air_quality)
st.write("**Traffic Congestion:**", traffic)
st.write("**Accident Severity:**", accident_severity)

# ==========================================================
# SCORE BREAKDOWN
# ==========================================================

st.subheader("Risk Score Breakdown")

st.write(f"Air Quality Score: **{air_score:.2f}/100**")
st.write(f"Traffic Score: **{traffic_score:.2f}/100**")
st.write(f"Accident Score: **{accident_score:.2f}/100**")