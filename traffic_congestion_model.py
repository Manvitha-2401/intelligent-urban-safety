import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv(
    r"data\TRAFFIC\Metro_Interstate_Traffic_Volume.csv"
)

print("Dataset shape:", data.shape)


# =========================================================
# 2. CREATE TRAFFIC CONGESTION CLASSES
# =========================================================

data["congestion"] = pd.cut(
    data["traffic_volume"],
    bins=[-1, 1193, 4933, 7280],
    labels=["Low", "Medium", "High"]
)

print("\nCongestion distribution:")
print(data["congestion"].value_counts())


# =========================================================
# 3. DATE-TIME FEATURE ENGINEERING
# =========================================================

data["date_time"] = pd.to_datetime(data["date_time"])

data["hour"] = data["date_time"].dt.hour
data["day_of_week"] = data["date_time"].dt.dayofweek
data["month"] = data["date_time"].dt.month

data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)

data["rush_hour"] = data["hour"].isin(
    [7, 8, 9, 16, 17, 18]
).astype(int)


# =========================================================
# 4. HANDLE MISSING VALUES
# =========================================================

data["holiday"] = data["holiday"].fillna("None")

data["rain_1h"] = data["rain_1h"].fillna(0)
data["snow_1h"] = data["snow_1h"].fillna(0)

data["weather_main"] = data["weather_main"].fillna("Unknown")
data["weather_description"] = data[
    "weather_description"
].fillna("Unknown")


# =========================================================
# 5. SELECT FEATURES
# =========================================================

features = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "rush_hour"
]

X = data[features]
y = data["congestion"]


# =========================================================
# 6. ENCODE TARGET
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget classes:")
print(label_encoder.classes_)


# =========================================================
# 7. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 8. FEATURE TYPES
# =========================================================

categorical_features = [
    "holiday",
    "weather_main",
    "weather_description"
]

numeric_features = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "rush_hour"
]


# =========================================================
# 9. PREPROCESSING
# =========================================================

preprocessor_scaled = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# =========================================================
# 10. DEFINE MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "preprocessing",
            preprocessor_scaled
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "XGBoost": Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42
            )
        )
    ]),

    "LightGBM": Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
        )
    ])
}


# =========================================================
# 11. TRAIN AND EVALUATE MODELS
# =========================================================

results = []
trained_models = {}

print("\n" + "=" * 70)
print("TRAFFIC CONGESTION MODEL RESULTS")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    trained_models[name] = model

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print(f"{name}: Accuracy = {accuracy * 100:.2f}%")

# =========================================================
# 12. RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# 13. BEST MODEL
# =========================================================

best_model = results_df.iloc[0]

best_algorithm = best_model["Algorithm"]
best_pipeline = trained_models[best_algorithm]

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_pipeline,
    "models/traffic_congestion_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/traffic_label_encoder.pkl"
)

print("\nBest traffic model saved successfully!")
print("Algorithm:", best_algorithm)
print("Model file: models/traffic_congestion_model.pkl")
print("Label encoder: models/traffic_label_encoder.pkl")


# =========================================================
# 14. SAVE RESULTS
# =========================================================

results_df.to_csv(
    "traffic_algorithm_comparison_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "traffic_algorithm_comparison_results.csv"
)


# =========================================================
# 15. CREATE ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.xlabel(
    "Machine Learning Algorithms"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Traffic Congestion Prediction - Algorithm Comparison"
)

plt.xticks(
    rotation=20
)

plt.ylim(
    0,
    100
)

for i, value in enumerate(
    results_df["Accuracy"]
):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "traffic_algorithm_accuracy.png",
    dpi=300
)

plt.show()

print(
    "\nGraph saved as "
    "traffic_algorithm_accuracy.png"
)