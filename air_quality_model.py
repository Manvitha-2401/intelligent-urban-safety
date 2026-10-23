import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    r"data\POPULATION\updated_pollution_dataset.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop("Air Quality", axis=1)

y = data["Air Quality"]


# ==========================================
# 3. ENCODE TARGET LABELS
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nAir Quality Classes:")
print(label_encoder.classes_)


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 5. CREATE LIGHTGBM MODEL
# ==========================================

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining LightGBM model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("       AIR QUALITY MODEL RESULTS")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ==========================================
# 9. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    r"models\lightgbm_air_quality.pkl"
)

joblib.dump(
    label_encoder,
    r"models\air_quality_label_encoder.pkl"
)

print("\nModel saved successfully!")
print("Saved as: models\\lightgbm_air_quality.pkl")

print("Label encoder saved successfully!")
print("Saved as: models\\air_quality_label_encoder.pkl")


# ==========================================
# 10. TEST WITH USER INPUT
# ==========================================

print("\n==========================================")
print("       AIR QUALITY PREDICTION")
print("==========================================")

temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
pm25 = float(input("Enter PM2.5: "))
pm10 = float(input("Enter PM10: "))
no2 = float(input("Enter NO2: "))
so2 = float(input("Enter SO2: "))
co = float(input("Enter CO: "))
industrial = float(
    input("Enter Proximity to Industrial Areas: ")
)
population = float(
    input("Enter Population Density: ")
)


# Create input dataframe
input_data = pd.DataFrame(
    [[
        temperature,
        humidity,
        pm25,
        pm10,
        no2,
        so2,
        co,
        industrial,
        population
    ]],
    columns=[
        "Temperature",
        "Humidity",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "Proximity_to_Industrial_Areas",
        "Population_Density"
    ]
)


# Predict
prediction = model.predict(input_data)

predicted_label = label_encoder.inverse_transform(
    prediction
)[0]


print("\n==========================================")
print("Predicted Air Quality:", predicted_label)
print("==========================================")