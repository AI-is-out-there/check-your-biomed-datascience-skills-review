"""
Задача 1: табличная классификация (риск метаболического / сердечно-сосудистого заболевания).
Модель: FFN — 1 скрытый слой, 32 нейрона, ReLU, 150 эпох (PyTorch).

Запуск из корня склонированного репозитория:
    python your-sollution/task1_solution.py
Результат: outputs/task1_predictions.csv (patient_id, prediction) + графики в outputs/task1_figures/
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

SEED = 42
EPOCHS = 150
HIDDEN = 32
LR = 1e-3
IQR_K = 1.5          # множитель для правила IQR
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1-2. Загрузка
TRAIN_PATH = "dataset/task1/train.csv"
TEST_PATH = "dataset/task1/test_features.csv"
OUT_PATH = "outputs/task1_predictions.csv"
FIG_DIR = "outputs/task1_figures"
os.makedirs("outputs", exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print("train:", train.shape, "| test:", test.shape)

TARGET = "prediction"
NON_FEATURES = ["patient_id", "prediction", "Статус_глюкозы", "Доклинический_риск"]
FEATURES = [c for c in train.columns if c not in NON_FEATURES]
print("Признаков:", len(FEATURES))
print("Баланс классов:\n", train[TARGET].value_counts())

# 3. Выбросы (IQR), только в train

continuous = [c for c in FEATURES if train[c].nunique() > 2]


def iqr_mask(df, cols, k=IQR_K):
    keep = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1

        keep &= df[c].isna() | df[c].between(q1 - k * iqr, q3 + k * iqr)
    return keep


mask = iqr_mask(train, continuous)
clean = train[mask].reset_index(drop=True)
print(f"После удаления выбросов: {len(clean)} из {len(train)} записей")

X = clean[FEATURES]
y = clean[TARGET].values

#  6. train / val / test (60/20/20, стратифицированно)
X_tmp, X_test_h, y_tmp, y_test_h = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
print("train/val/test:", len(X_tr), len(X_val), len(X_test_h))

#  4-5. Пропуски средним + StandardScaler
imputer = SimpleImputer(strategy="mean").fit(X_tr)
scaler = StandardScaler().fit(imputer.transform(X_tr))


def prep(df):
    return scaler.transform(imputer.transform(df)).astype(np.float32)


Xtr_t = torch.tensor(prep(X_tr))
Xval_t = torch.tensor(prep(X_val))
Xte_t = torch.tensor(prep(X_test_h))
ytr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

# 7. Модель: 1 скрытый слой, 32 нейрона, ReLU
model = nn.Sequential(
    nn.Linear(len(FEATURES), HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 1),
)

pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {"train_loss": [], "val_auc": []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(Xtr_t), ytr_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xval_t)).numpy().ravel()
    history["train_loss"].append(loss.item())
    history["val_auc"].append(roc_auc_score(y_val, val_prob))
    if epoch % 25 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val AUC {history['val_auc'][-1]:.4f}")


def predict_proba(x_tensor):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_tensor)).numpy().ravel()


# Оценка на отложенной выборке
THRESHOLD = 0.5
p_test = predict_proba(Xte_t)
pred_test = (p_test >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test_h, pred_test).ravel()
print("\n=== Отложенная выборка ===")
print(f"Accuracy   {accuracy_score(y_test_h, pred_test):.3f}")
print(f"Recall     {recall_score(y_test_h, pred_test):.3f}")
print(f"Specificity{tn / (tn + fp):.3f}")
print(f"Precision  {precision_score(y_test_h, pred_test, zero_division=0):.3f}")
print(f"F1         {f1_score(y_test_h, pred_test):.3f}")
print(f"ROC-AUC    {roc_auc_score(y_test_h, p_test):.4f}")
print("Confusion matrix [[TN FP],[FN TP]]:\n", np.array([[tn, fp], [fn, tp]]))

#  Графики
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(history["train_loss"]); ax[0].set_title("Train loss"); ax[0].set_xlabel("epoch")
ax[1].plot(history["val_auc"]); ax[1].set_title("Val ROC-AUC"); ax[1].set_xlabel("epoch")
fpr, tpr, _ = roc_curve(y_test_h, p_test)
ax[2].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test_h, p_test):.3f}")
ax[2].plot([0, 1], [0, 1], "--", color="gray"); ax[2].set_title("ROC (test)"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/training_and_roc.png", dpi=120); plt.close()

plt.figure(figsize=(4, 4))
plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
    plt.text(j, i, v, ha="center", va="center")
plt.xticks([0, 1], ["pred 0", "pred 1"]); plt.yticks([0, 1], ["true 0", "true 1"])
plt.title("Confusion matrix (test)"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=120); plt.close()

#  8-9. Предсказания для test_features.csv
# Препроцессинг — тот же самый (fit на train). Ни одну строку не удаляем и не переставляем.
Xsub_t = torch.tensor(prep(test[FEATURES]))
p_sub = predict_proba(Xsub_t)
submission = pd.DataFrame({
    "patient_id": test["patient_id"].values,
    "prediction": (p_sub >= THRESHOLD).astype(int),
})
assert len(submission) == len(test) and submission.notna().all().all()
submission.to_csv(OUT_PATH, index=False)
print(f"\nСохранено: {OUT_PATH} ({len(submission)} строк, "
      f"доля класса 1: {submission['prediction'].mean():.2%})")
