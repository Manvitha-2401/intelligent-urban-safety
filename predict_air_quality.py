import joblib
import pandas as pd

# Load trained model and label encoder
model = joblib.load("models/lightgbm_air_quality.pkl")
label_encoder = joblib.load("models/air_quality_label_encoder.pkl")

print("\n🌫️ AIR QUALITY PREDICTION")
print("----------------------------")

# Get user inputs
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
pm25 = float(input("PM2.5: "))
pm10 = float(input("PM10: "))
no2 = float(input("NO2: "))
so2 = float(input("SO2: "))
co = float(input("CO: "))
industrial_distance = float(input("Proximity to Industrial Area: "))
population_density = int(input("Population Density: "))

# Create input dataframe
input_data = pd.DataFrame([[
    temperature,
    humidity,
    pm25,
    pm10,
    no2,
    so2,
    co,
    industrial_distance,
    population_density
]], columns=[
    "Temperature",
    "Humidity",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Proximity_to_Industrial_Areas",
    "Population_Density"
])

# Make prediction
prediction = model.predict(input_data)

# Convert prediction back to original category
air_quality = label_encoder.inverse_transform(prediction)[0]

print("\n----------------------------")
print("Predicted Air Quality:", air_quality)
print("----------------------------")
