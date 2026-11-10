"""
Задание 3.1: классификация гистопатологических изображений
"""
import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# настройки
SEED = 42
MAX_EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 128
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "dataset" / "task3_1" / "train.npz"
TEST_PATH = ROOT / "dataset" / "task3_1" / "test_features.npz"
OUT_PATH = ROOT / "outputs" / "task3_1_predictions.csv"
FIG_DIR = ROOT / "outputs" / "task3_1_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. данные
train = np.load(TRAIN_PATH, allow_pickle=True)
test = np.load(TEST_PATH, allow_pickle=True)
X, y = train["images"], train["prediction"].astype(np.int64)
label_names = [str(n) for n in train["label_names"]]
X_test, test_ids = test["images"], test["image_id"]
print("train:", X.shape, "test:", X_test.shape, "device:", device)
print("баланс классов:", dict(zip(label_names, np.bincount(y))))

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# примеры изображений
fig, axes = plt.subplots(9, 6, figsize=(9, 13))
for c in range(9):
    for j, i in enumerate(np.where(y == c)[0][:6]):
        axes[c, j].imshow(X[i])
        axes[c, j].axis("off")
    axes[c, 0].set_title(f"{c}: {label_names[c]}", fontsize=8, loc="left")
plt.tight_layout()
plt.savefig(FIG_DIR / "samples.png", dpi=110)
plt.close()


# 2. предобработка
def to_tensor(imgs):
    """uint8 (N,28,28,3) -> float32 (N,3,28,28), в диапазоне [-1, 1]."""
    t = torch.from_numpy(imgs).float().div(255.0).permute(0, 3, 1, 2)
    return (t - 0.5) / 0.5


Xt_tr, Xt_val, Xt_test = to_tensor(X_tr), to_tensor(X_val), to_tensor(X_test)
yt_tr, yt_val = torch.from_numpy(y_tr), torch.from_numpy(y_val)
train_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=256)
val_loader = DataLoader(TensorDataset(Xt_val, yt_val), batch_size=256)


def augment(batch):
    """Случайные отражения и повороты на 90 градусов: у гистологии нет 'верха'."""
    if random.random() < 0.5:
        batch = torch.flip(batch, dims=[3])
    if random.random() < 0.5:
        batch = torch.flip(batch, dims=[2])
    return torch.rot90(batch, random.randint(0, 3), dims=[2, 3])


# 3. модели
class FFN(nn.Module):
    """Полносвязная сеть: картинка 3*28*28=2352 -> 256 -> 128 -> 9 (пространственная структура теряется)."""

    def __init__(self, n_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """3 блока Conv-BN-ReLU-MaxPool (28->14->7->3), затем полносвязная голова."""

    def __init__(self, n_classes=9):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


counts = np.bincount(y_tr, minlength=9)
class_w = torch.tensor(counts.sum() / (9 * counts), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_w)


def evaluate(model, loader):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            losses.append(criterion(out, yb).item() * len(xb))
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return (sum(losses) / len(trues), accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro"), preds)


def fit(model, name, use_augment):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    hist = {k: [] for k in ["tr_loss", "val_loss", "tr_f1", "val_f1", "val_acc"]}
    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            if use_augment:
                xb = augment(xb)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        tr_loss, _, tr_f1, _ = evaluate(model, train_eval_loader)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader)
        for k, v in zip(hist, [tr_loss, val_loss, tr_f1, val_f1, val_acc]):
            hist[k].append(v)
        sched.step(val_f1)
        print(f"[{name}] epoch {epoch:02d} | loss {tr_loss:.3f}/{val_loss:.3f} | F1 {tr_f1:.3f}/{val_f1:.3f} | val acc {val_acc:.3f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[{name}] ранняя остановка")
                break
    model.load_state_dict(best_state)
    return model, hist


results = {}
for name, model, aug in [("FFN", FFN(), False), ("CNN", SmallCNN(), True)]:
    model, hist = fit(model, name, aug)
    _, acc, f1, pred = evaluate(model, val_loader)
    results[name] = dict(model=model, hist=hist, acc=acc, f1=f1, pred=pred)
    print(f"\n[{name}] VAL accuracy = {acc:.4f}, macro F1 = {f1:.4f}")
    print(classification_report(y_val, pred, target_names=label_names, digits=3))

    # графики обучения
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title(f"{name}: Loss"); ax[0].set_xlabel("эпоха"); ax[0].legend()
    ax[1].plot(hist["tr_f1"], label="train macro F1"); ax[1].plot(hist["val_f1"], label="val macro F1")
    ax[1].plot(hist["val_acc"], "--", label="val accuracy")
    ax[1].set_title(f"{name}: качество"); ax[1].set_xlabel("эпоха"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name.lower()}_training_curves.png", dpi=110); plt.close()

    # матрица ошибок
    plt.figure(figsize=(9, 7))
    sns.heatmap(confusion_matrix(y_val, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("предсказано"); plt.ylabel("истина"); plt.title(f"{name}: матрица ошибок (валидация)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name.lower()}_confusion_matrix.png", dpi=110); plt.close()

# 4. выбор лучшей модели и примеры ошибок
best_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_name]
print(f"\nИтог: FFN F1={results['FFN']['f1']:.4f} | CNN F1={results['CNN']['f1']:.4f} -> лучшая: {best_name}")

wrong = np.where(best["pred"] != y_val)[0][:12]
if len(wrong):
    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    for ax_, i in zip(axes.ravel(), wrong):
        ax_.imshow(X_val[i])
        ax_.set_title(f"истина: {y_val[i]}\nпредск.: {best['pred'][i]}", fontsize=8)
        ax_.axis("off")
    plt.tight_layout(); plt.savefig(FIG_DIR / "errors.png", dpi=110); plt.close()

# 5. предсказания для теста (метки теста не используются)
model = best["model"].eval()
with torch.no_grad():
    test_pred = torch.cat([model(Xt_test[i:i + 256].to(device)).argmax(1).cpu()
                           for i in range(0, len(Xt_test), 256)]).numpy()

pred_df = pd.DataFrame({"image_id": test_ids, "prediction": test_pred.astype(int)})
assert len(pred_df) == len(X_test) and pred_df["prediction"].between(0, 8).all()
pred_df.to_csv(OUT_PATH, index=False)
print("сохранено:", OUT_PATH, pred_df.shape)
print("распределение предсказаний:", pred_df["prediction"].value_counts().sort_index().to_dict())
