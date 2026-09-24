from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

repo_root = Path.cwd()
if repo_root.name in {"vanilla-code", "your-sollution", "your_solution"}:
    repo_root = repo_root.parent

train = pd.read_csv(repo_root / "dataset" / "task1" / "train.csv")
test = pd.read_csv(repo_root / "dataset" / "task1" / "test_features.csv")

feature_cols = [c for c in train.columns if c not in ("patient_id", "prediction")]

X_train = train[feature_cols]
y_train = train["prediction"]
X_test = test[feature_cols]

model = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=3,
    learning_rate=0.05,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test).astype(int)

submission = pd.DataFrame({"patient_id": test["patient_id"], "prediction": predictions})

out_dir = repo_root / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
submission.to_csv(out_dir / "task1_predictions.csv", index=False)

print(f"Saved {len(submission)} predictions")