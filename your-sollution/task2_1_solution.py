"""
Задание 2.1: классификация сердечных сокращений по ЭКГ

"""
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Настройки
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 10          # ранняя остановка: ждём улучшения val macro F1 столько эпох
BATCH_SIZE = 128
LR = 1e-3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "dataset" / "task2_1" / "train.csv"
TEST_PATH = ROOT / "dataset" / "task2_1" / "test_features.csv"
OUT_PATH = ROOT / "outputs" / "task2_1_predictions.csv"
FIG_DIR = ROOT / "outputs" / "task2_1_figures"
os.makedirs(FIG_DIR, exist_ok=True)

LABELS = ["0: N (норма)", "1: R-on-T PVC", "2: PVC", "3: SP/EB", "4: UB"]

# 1. Загрузка и знакомство с данными
df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
signal_cols = [c for c in df.columns if c.startswith("t_")]      # t_000 ... t_139
print("train:", df.shape, "| test:", test_df.shape, "| точек в сигнале:", len(signal_cols))
print("Распределение классов:\n", df["prediction"].value_counts().sort_index())

X = df[signal_cols].to_numpy(dtype=np.float32)
y = df["prediction"].to_numpy(dtype=np.int64)

# График 1: по одному сигналу каждого класса
fig, axes = plt.subplots(1, 5, figsize=(20, 3.5), sharey=True)
for k, ax in enumerate(axes):
    ax.plot(X[np.where(y == k)[0][0]], lw=2)
    ax.set_title(LABELS[k]); ax.set_xlabel("время")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/1_class_examples.png", dpi=120); plt.close()

#  2. train / val / test (70/15/15, стратифицированно)
# Стратификация сохраняет долю каждого класса (особенно редкого класса 4) во всех частях.
X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.1765, stratify=y_tv, random_state=SEED)
print("train/val/test:", len(X_tr), len(X_va), len(X_te))

# 3. Нормализация
scaler = StandardScaler().fit(X_tr)
X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)


def make_loader(features, targets=None, shuffle=False):
    xt = torch.tensor(features[:, None, :], dtype=torch.float32)
    ds = TensorDataset(xt) if targets is None else TensorDataset(xt, torch.tensor(targets, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = make_loader(X_tr, y_tr, shuffle=True)
val_loader = make_loader(X_va, y_va)
test_loader = make_loader(X_te, y_te)


# 4. Модель: 1D CNN
class ECGClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),   # 140 -> 70
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),  # 70 -> 35
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                                                   # 35 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Дисбаланс: вес класса = N / (число_классов * число_объектов_класса).
# Редкий класс получает большой вес, следовательно ошибка на нём штрафуется сильнее.
counts = np.bincount(y_tr, minlength=5)
weights = torch.tensor(len(y_tr) / (5 * counts), dtype=torch.float32, device=device)
print("Веса классов:", weights.cpu().numpy().round(2))

model = ECGClassifier(5).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# 5. Обучение + ранняя остановка по val macro F1
def run_epoch(loader, train_mode):
    model.train(train_mode)
    total_loss, preds, targets = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(yb.cpu().numpy())
    return total_loss / len(loader.dataset), f1_score(targets, preds, average="macro")


history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
best_f1, best_state, wait = -1.0, None, 0
for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_f1 = run_epoch(train_loader, True)
    va_loss, va_f1 = run_epoch(val_loader, False)
    for k, v in zip(history, (tr_loss, va_loss, tr_f1, va_f1)):
        history[k].append(v)
    print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} F1 {tr_f1:.4f} | val loss {va_loss:.4f} F1 {va_f1:.4f}")
    if va_f1 > best_f1:
        best_f1, wait = va_f1, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Ранняя остановка.")
            break
model.load_state_dict(best_state)       # возвращаем лучшую по val F1 версию модели
print(f"Лучший val macro F1: {best_f1:.4f}")

# График 2: loss и macro F1
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history["train_f1"], label="train"); ax[1].plot(history["val_f1"], label="val")
ax[1].set_title("Macro F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/2_loss_f1.png", dpi=120); plt.close()

# 6. Итоговая оценка на отложенном test
model.eval()
te_pred, te_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        te_pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        te_true.extend(yb.numpy())
print("\n=== Test ===")
print(f"Accuracy {accuracy_score(te_true, te_pred):.4f} | macro F1 {f1_score(te_true, te_pred, average='macro'):.4f}")
print(classification_report(te_true, te_pred, labels=range(5), target_names=LABELS, zero_division=0))

cm = confusion_matrix(te_true, te_pred, labels=range(5))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Предсказано"); plt.ylabel("Истинный класс"); plt.title("Confusion matrix (test)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/3_confusion_matrix.png", dpi=120); plt.close()

# 7. Предсказания для test_features.csv
X_sub = scaler.transform(test_df[signal_cols].to_numpy(dtype=np.float32))
sub_pred = []
with torch.no_grad():
    for (xb,) in make_loader(X_sub):
        sub_pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())

submission = pd.DataFrame({"heartbeat_id": test_df["heartbeat_id"].values,
                           "prediction": np.array(sub_pred, dtype=int)})
assert len(submission) == len(test_df) and submission.notna().all().all()
assert submission["prediction"].between(0, 4).all()
submission.to_csv(OUT_PATH, index=False)
print(f"\nСохранено: {OUT_PATH} ({len(submission)} строк)")
print(submission["prediction"].value_counts().sort_index())
