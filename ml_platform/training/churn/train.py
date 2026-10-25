from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=563)

X = dataset.data.features
y = dataset.data.targets.iloc[:, 0]

print("Shape:", X.shape)
print("Target distribution:")
print(y.value_counts(normalize=True))
print(X.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)
print("Training complete")


from sklearn.metrics import classification_report, roc_auc_score

predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

# Sanity check:
sample = X_test.head(5)
print(sample.to_string())
print("Predictions:", pipeline.predict(sample))
print("Churn probabilities:", pipeline.predict_proba(sample)[:, 1].round(3))
print("Actual labels:", y_test.head(5).tolist())

# Checking which features matter most?
coefs = pipeline.named_steps["model"].coef_[0]
for name, coef in sorted(zip(X.columns, coefs), key=lambda p: abs(p[1]), reverse=True):
    print(f"{name:30s} {coef:+.3f}")


import json
from pathlib import Path

import joblib

artifact_dir = Path("ml_platform/training/churn/models")
artifact_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, artifact_dir / "churn_pipeline.joblib")

(artifact_dir / "feature_names.json").write_text(
    json.dumps(X.columns.tolist(), indent=2)
)
print("Saved artifact to", artifact_dir)
