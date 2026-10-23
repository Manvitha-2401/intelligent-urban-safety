import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load dataset
data = pd.read_csv(r"data\POPULATION\updated_pollution_dataset.csv")

# Features and target
X = data.drop("Air Quality", axis=1)
y = LabelEncoder().fit_transform(data["Air Quality"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        verbosity=-1,
        random_state=42
    )
}

print("\n==============================================")
print("   AIR QUALITY ALGORITHM COMPARISON")
print("==============================================")

results = []

for name, model in models.items():

    # Use scaled data only for Logistic Regression
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction) * 100

    results.append([name, accuracy])

    print(f"{name}: {accuracy:.2f}%")

print("==============================================")