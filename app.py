import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Urban Safety Prediction System",
    page_icon="🌆",
    layout="wide"
)

st.title("🌆 Intelligent Urban Safety Prediction System")
st.write(
    "Predict air quality, traffic congestion, accident severity "
    "and overall urban risk using trained machine learning models."
)

# --------------------------------------------------
# 2. LOAD TRAINED MODELS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_models():
    air_model = joblib.load(MODEL_DIR / "lightgbm_air_quality.pkl")
    air_encoder = joblib.load(
        MODEL_DIR / "air_quality_label_encoder.pkl"
    )

    traffic_model = joblib.load(
        MODEL_DIR / "traffic_congestion_model.pkl"
    )
    traffic_encoder = joblib.load(
        MODEL_DIR / "traffic_label_encoder.pkl"
    )

    accident_model = joblib.load(
        MODEL_DIR / "accident_severity_model.pkl"
    )

    return (
        air_model,
        air_encoder,
        traffic_model,
        traffic_encoder,
        accident_model
    )


try:
    (
        air_model,
        air_encoder,
        traffic_model,
        traffic_encoder,
        accident_model
    ) = load_models()

except Exception as e:
    st.error("Could not load the trained models.")
    st.error(str(e))
    st.info(
        "Check that all five required .pkl files exist in your models folder."
    )
    st.stop()


# --------------------------------------------------
# 3. INPUT FEATURES
# --------------------------------------------------

st.header("Enter Environmental and Road Conditions")

st.caption(
    "Enter the available conditions for the location you want to assess. "
    "The predictions depend on the quality and relevance of these inputs."
)

with st.expander("🌫️ Air Quality Inputs", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:
        temperature = st.number_input(
            "Temperature (°C)", value=25.0
        )
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0,
            max_value=100.0, value=60.0
        )
        pm25 = st.number_input("PM2.5", min_value=0.0, value=25.0)

    with c2:
        pm10 = st.number_input("PM10", min_value=0.0, value=50.0)
        no2 = st.number_input("NO2", min_value=0.0, value=20.0)
        so2 = st.number_input("SO2", min_value=0.0, value=10.0)

    with c3:
        co = st.number_input("CO", min_value=0.0, value=1.0)
        industrial_distance = st.number_input(
            "Proximity to Industrial Areas",
            min_value=0.0, value=5.0
        )
        population_density = st.number_input(
            "Population Density",
            min_value=0, value=500, step=10
        )


with st.expander("🚗 Traffic Congestion Inputs", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:
        holiday = st.selectbox(
            "Holiday",
            [
                "None", "Columbus Day", "Veterans Day",
                "Thanksgiving Day", "Christmas Day",
                "New Years Day", "Washingtons Birthday",
                "Memorial Day", "Independence Day",
                "Labor Day", "State Fair"
            ]
        )

        traffic_temp = st.number_input(
            "Traffic Dataset Temperature (Kelvin)",
            value=273.15
        )

        rain = st.number_input(
            "Rain in Last Hour", min_value=0.0, value=0.0
        )

    with c2:
        snow = st.number_input(
            "Snow in Last Hour", min_value=0.0, value=0.0
        )

        clouds = st.slider(
            "Cloud Coverage (%)", 0, 100, 40
        )

        weather_main = st.selectbox(
            "Main Weather",
            [
                "Clear", "Clouds", "Rain", "Drizzle",
                "Mist", "Snow", "Haze", "Fog",
                "Thunderstorm", "Smoke", "Squall"
            ]
        )

    with c3:
        weather_description = st.selectbox(
            "Weather Description",
            [
                "sky is clear",
                "few clouds",
                "scattered clouds",
                "broken clouds",
                "overcast clouds",
                "light rain",
                "moderate rain",
                "heavy intensity rain",
                "mist",
                "fog",
                "light snow",
                "snow",
                "thunderstorm"
            ]
        )

        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x]
        )
        month = st.slider("Month", 1, 12, 6)

    is_weekend = int(day_of_week >= 5)
    rush_hour = int(hour in [7, 8, 9, 16, 17, 18])


with st.expander("🚧 Accident Severity Inputs", expanded=True):

    st.write("Enter the location, weather and road conditions.")

    c1, c2, c3 = st.columns(3)

    with c1:
        start_lat = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            value=13.0827, format="%.6f"
        )
        start_lng = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=80.2707, format="%.6f"
        )
        distance = st.number_input(
            "Distance (miles)", min_value=0.0, value=0.1
        )
        acc_temp = st.number_input(
            "Temperature (°F)", value=77.0
        )
        wind_chill = st.number_input(
            "Wind Chill (°F)", value=77.0
        )
        acc_humidity = st.slider(
            "Accident Weather Humidity (%)", 0, 100, 60
        )
        pressure = st.number_input(
            "Pressure (in)", min_value=0.0, value=29.9
        )
        visibility = st.number_input(
            "Visibility (miles)", min_value=0.0, value=10.0
        )
        wind_speed = st.number_input(
            "Wind Speed (mph)", min_value=0.0, value=5.0
        )
        precipitation = st.number_input(
            "Precipitation (in)", min_value=0.0, value=0.0
        )

    with c2:
        accident_hour = st.slider(
            "Accident Hour", 0, 23, 12
        )
        accident_day = st.selectbox(
            "Accident Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x]
        )
        accident_month = st.slider(
            "Accident Month", 1, 12, 6
        )
        accident_weekend = int(accident_day >= 5)

        st.caption("Select road conditions in the third column.")

    # Obtain the categorical options learned during training.
    # This helps preserve the categories expected by the saved model.

    try:
        accident_preprocessor = accident_model.named_steps["preprocessor"]

        categorical_transformer = next(
            transformer
            for name, transformer, columns
            in accident_preprocessor.transformers_
            if name == "categorical"
        )

        categorical_columns = list(
            next(
                columns
                for name, transformer, columns
                in accident_preprocessor.transformers_
                if name == "categorical"
            )
        )

        category_map = dict(
            zip(
                categorical_columns,
                categorical_transformer.categories_
            )
        )

    except Exception as e:
        st.error(
            "Could not read the accident model's categorical features. "
            "Please share the accident_model.py training code."
        )
        st.error(str(e))
        st.stop()

    with c3:

        if "Weather_Condition" in category_map:
            weather_options = list(category_map["Weather_Condition"])
        else:
            weather_options = ["Clear"]

        weather_default = next(
            (
                i for i, value in enumerate(weather_options)
                if str(value).lower() in ["clear", "fair"]
            ),
            0
        )

        accident_weather = st.selectbox(
            "Accident Weather Condition",
            weather_options,
            index=weather_default
        )

        road_features = [
            "Amenity", "Bump", "Crossing", "Give_Way",
            "Junction", "No_Exit", "Railway", "Roundabout",
            "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
        ]

        road_values = {}

        for feature in road_features:
            options = list(category_map[feature])

            false_index = next(
                (
                    i for i, value in enumerate(options)
                    if str(value).lower() == "false"
                ),
                0
            )

            road_values[feature] = st.selectbox(
                feature.replace("_", " "),
                options,
                index=false_index,
                key="road_" + feature
            )


# --------------------------------------------------
# 4. PREDICTION BUTTON
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "Predict Urban Risk",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# 5. RUN ALL THREE MODELS
# --------------------------------------------------

if predict_button:

    try:
        # Air quality prediction
        air_input = pd.DataFrame(
            [[
                temperature, humidity, pm25, pm10, no2, so2, co,
                industrial_distance, population_density
            ]],
            columns=[
                "Temperature", "Humidity", "PM2.5", "PM10", "NO2",
                "SO2", "CO", "Proximity_to_Industrial_Areas",
                "Population_Density"
            ]
        )

        air_prediction = air_model.predict(air_input)
        air_result = str(
            air_encoder.inverse_transform(
                np.asarray(air_prediction).astype(int)
            )[0]
        )

        # Traffic congestion prediction
        traffic_input = pd.DataFrame(
            [[
                holiday, traffic_temp, rain, snow, clouds,
                weather_main, weather_description, hour,
                day_of_week, month, is_weekend, rush_hour
            ]],
            columns=[
                "holiday", "temp", "rain_1h", "snow_1h",
                "clouds_all", "weather_main",
                "weather_description", "hour", "day_of_week",
                "month", "is_weekend", "rush_hour"
            ]
        )

        traffic_prediction = traffic_model.predict(traffic_input)

        traffic_result = str(
            traffic_encoder.inverse_transform(
                np.asarray(traffic_prediction).astype(int)
            )[0]
        )

        # Accident severity prediction
        accident_data = {
            "Start_Lat": start_lat,
            "Start_Lng": start_lng,
            "Distance(mi)": distance,
            "Temperature(F)": acc_temp,
            "Wind_Chill(F)": wind_chill,
            "Humidity(%)": acc_humidity,
            "Pressure(in)": pressure,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Condition": accident_weather,
            "Amenity": road_values["Amenity"],
            "Bump": road_values["Bump"],
            "Crossing": road_values["Crossing"],
            "Give_Way": road_values["Give_Way"],
            "Junction": road_values["Junction"],
            "No_Exit": road_values["No_Exit"],
            "Railway": road_values["Railway"],
            "Roundabout": road_values["Roundabout"],
            "Station": road_values["Station"],
            "Stop": road_values["Stop"],
            "Traffic_Calming": road_values["Traffic_Calming"],
            "Traffic_Signal": road_values["Traffic_Signal"],
            "Hour": accident_hour,
            "Day_of_Week": accident_day,
            "Month": accident_month,
            "Is_Weekend": accident_weekend
        }

        accident_input = pd.DataFrame([accident_data])

        accident_prediction = int(
            np.asarray(accident_model.predict(accident_input))[0]
        )

        # Training mapped severity 1–4 to encoded values 0–3.
        accident_result = accident_prediction + 1

        # --------------------------------------------------
        # 6. CALCULATE OVERALL URBAN RISK SCORE
        # --------------------------------------------------

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

        air_score = air_scores.get(air_result, 50)
        traffic_score = traffic_scores.get(traffic_result, 50)

        accident_score = (
            (accident_result - 1) / 3
        ) * 100

        overall_score = (
            0.35 * air_score
            + 0.30 * traffic_score
            + 0.35 * accident_score
        )

        if overall_score <= 33:
            risk_level = "Low"
        elif overall_score <= 66:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # --------------------------------------------------
        # 7. DISPLAY RESULTS
        # --------------------------------------------------

        st.divider()
        st.header("Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Air Quality", air_result)
        c2.metric("Traffic Congestion", traffic_result)
        c3.metric("Predicted Accident Severity", str(accident_result))

        st.subheader("Overall Urban Risk")

        st.metric(
            "Urban Risk Score",
            f"{overall_score:.2f} / 100",
            help="Weighted score from the three model predictions."
        )

        st.progress(int(round(overall_score)))

        st.subheader(f"Risk Level: {risk_level}")

        if risk_level == "Low":
            st.success("The combined model score indicates relatively low risk.")
        elif risk_level == "Moderate":
            st.warning("The combined model score indicates moderate risk.")
        else:
            st.error("The combined model score indicates high risk.")

        st.subheader("Risk Score Breakdown")

        result_table = pd.DataFrame({
            "Component": [
                "Air Quality",
                "Traffic Congestion",
                "Accident Severity"
            ],
            "Predicted Result": [
                air_result,
                traffic_result,
                f"Severity {accident_result}"
            ],
            "Risk Score": [
                round(air_score, 2),
                round(traffic_score, 2),
                round(accident_score, 2)
            ],
            "Weight (%)": [35, 30, 35]
        })

        st.dataframe(result_table, use_container_width=True)

        st.caption(
            "These are model-based estimates, not live sensor measurements. "
            "The overall score is a project-defined weighted index, "
            "not an official emergency or safety rating."
        )

    except Exception as e:
        st.error("A prediction could not be completed.")
        st.code(str(e))
        st.info(
            "Please share the full error message so we can correct "
            "the input format or model integration."
        )

        
# --------------------------------------------------
# 8. ALGORITHM PERFORMANCE COMPARISON
# --------------------------------------------------

st.divider()
st.header("Algorithm Performance Comparison")

comparison_files = {
    "Air Quality": BASE_DIR / "air_quality_algorithm_comparison_results.csv",
    "Traffic Congestion": BASE_DIR / "traffic_algorithm_comparison_results.csv",
    "Accident Severity": BASE_DIR / "accident_algorithm_comparison_results.csv",
}

for title, file_path in comparison_files.items():
    st.subheader(title)

    if file_path.exists():
        df = pd.read_csv(file_path)

        if "Algorithm" in df.columns and "Accuracy" in df.columns:
            df["Accuracy"] = pd.to_numeric(
                df["Accuracy"], errors="coerce"
            )
            df = df.dropna(subset=["Accuracy"])

            st.bar_chart(
                df.set_index("Algorithm")["Accuracy"]
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(
                f"Please check the columns in {file_path.name}"
            )
    else:
        st.warning(f"File not found: {file_path.name}")
