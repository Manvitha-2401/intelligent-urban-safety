import joblib
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

file_path = r"data\POPULATION\ACCIDENT\US_Accidents_March23.csv"

print("Loading accident dataset...")
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)


# ==========================================================
# 2. TAKE STRATIFIED SAMPLE
# ==========================================================

print("\nCreating stratified sample...")

sample_size = 500000

df_sample, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df["Severity"],
    random_state=42
)

df = df_sample.copy()

print("Sample shape:", df.shape)

print("\nSample severity distribution:")
print(df["Severity"].value_counts().sort_index())

print("\nSample severity percentage:")
print(
    (df["Severity"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)


# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

print("\nCreating features...")

# Convert Start_Time to datetime
df["Start_Time"] = pd.to_datetime(
    df["Start_Time"],
    errors="coerce"
)

# Time features
df["Hour"] = df["Start_Time"].dt.hour
df["Day_of_Week"] = df["Start_Time"].dt.dayofweek
df["Month"] = df["Start_Time"].dt.month
df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)


# ==========================================================
# 4. SELECT FEATURES
# ==========================================================

features = [
    # Location
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",

    # Weather
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",

    # Road/environment
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",

    # Time
    "Hour",
    "Day_of_Week",
    "Month",
    "Is_Weekend"
]

target = "Severity"

X = df[features].copy()

# XGBoost and LightGBM require class labels starting from 0
y = df[target].astype(int) - 1

print("\nEncoded severity classes:")
print(sorted(y.unique()))

# Convert Severity 1,2,3,4 → 0,1,2,3
severity_mapping = {
    1: 0,
    2: 1,
    3: 2,
    4: 3
}

y = y.map(severity_mapping)
X = df[features].copy()
y = df[target].astype(int) - 1
print("\nEncoded severity classes:")
print(sorted(y.unique()))



# ==========================================================
# 5. HANDLE MISSING VALUES
# ==========================================================

print("\nHandling missing values...")

numeric_features = [
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Hour",
    "Day_of_Week",
    "Month",
    "Is_Weekend"
]

categorical_features = [
    "Weather_Condition",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal"
]


# Fill numeric missing values
for column in numeric_features:

    if column == "Precipitation(in)":
        X[column] = X[column].fillna(0)

    else:
        X[column] = X[column].fillna(X[column].median())


# Fill categorical missing values
for column in categorical_features:
    X[column] = X[column].fillna("Unknown")


# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nDEBUG - Training labels:")
print(sorted(y_train.unique()))

print("DEBUG - Testing labels:")
print(sorted(y_test.unique()))


# ==========================================================
# 7. PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================================
# 8. MODELS
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
),

    "LightGBM": LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        objective="multiclass",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
}


# ==========================================================
# 9. TRAIN AND COMPARE MODELS
# ==========================================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    # Keep the trained pipeline for saving later
    trained_models[name] = pipeline

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\nResults:")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=[
                "Severity 1",
                "Severity 2",
                "Severity 3",
                "Severity 4"
            ],
            zero_division=0
        )
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1_Score": f1 * 100
    })


# ==========================================================
# 10. SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.round(2))


results_df.to_csv(
    "accident_algorithm_comparison_results.csv",
    index=False
)


# ==========================================================
# 11. FIND BEST MODEL
# ==========================================================

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest model based on accuracy:")
print(best_model)

# Save the best-performing trained pipeline
best_algorithm = best_model["Algorithm"]
best_pipeline = trained_models[best_algorithm]

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_pipeline,
    "models/accident_severity_model.pkl"
)

print("\nBest accident model saved successfully!")
print("Algorithm:", best_algorithm)
print("Accuracy:", round(best_model["Accuracy"], 2), "%")
print("Saved to: models/accident_severity_model.pkl")

# ==========================================================
# 12. CREATE ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.xlabel("Algorithm")
plt.ylabel("Accuracy (%)")
plt.title("Accident Severity Prediction - Algorithm Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "accident_algorithm_accuracy.png",
    dpi=300
)

plt.show()


print("\nResults saved successfully!")
print("CSV: accident_algorithm_comparison_results.csv")
print("Graph: accident_algorithm_accuracy.png")