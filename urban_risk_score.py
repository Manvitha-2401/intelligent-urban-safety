# ==========================================================
# OVERALL URBAN RISK SCORE
# ==========================================================

def calculate_risk_score(air_quality, traffic, accident_severity):

    # ------------------------------------------------------
    # 1. Convert Air Quality to 0-100
    # ------------------------------------------------------

    air_scores = {
        "Good": 0,
        "Moderate": 33.33,
        "Poor": 66.67,
        "Hazardous": 100
    }

    air_score = air_scores[air_quality]

    # ------------------------------------------------------
    # 2. Convert Traffic to 0-100
    # ------------------------------------------------------

    traffic_scores = {
        "Low": 0,
        "Medium": 50,
        "High": 100
    }

    traffic_score = traffic_scores[traffic]

    # ------------------------------------------------------
    # 3. Convert Accident Severity to 0-100
    # ------------------------------------------------------

    accident_score = ((accident_severity - 1) / 3) * 100

    # ------------------------------------------------------
    # 4. Calculate Overall Risk Score
    # ------------------------------------------------------

    overall_score = (
        air_score * 0.35
        + traffic_score * 0.30
        + accident_score * 0.35
    )

    return overall_score


# ==========================================================
# TEST
# ==========================================================

air_quality = "Moderate"
traffic = "High"
accident_severity = 3

risk_score = calculate_risk_score(
    air_quality,
    traffic,
    accident_severity
)

# Determine risk level
if risk_score <= 33:
    risk_level = "Low Risk"
elif risk_score <= 66:
    risk_level = "Moderate Risk"
else:
    risk_level = "High Risk"

print("======================================")
print("     OVERALL URBAN RISK SCORE")
print("======================================")

print("Air Quality:", air_quality)
print("Traffic:", traffic)
print("Accident Severity:", accident_severity)

print("--------------------------------------")
print(f"Overall Risk Score: {risk_score:.2f}/100")
print("Risk Level:", risk_level)