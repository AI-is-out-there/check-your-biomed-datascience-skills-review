"""
Задание 4.1: классификация генетических популяций с помощью графа сходства и GCN.
"""
import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# настройки
SEED = 42
K_NEIGHBORS = 5
N_PCA = 50
HIDDEN = 64
LR = 1e-2
MAX_EPOCHS = 300
PATIENCE = 40

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "dataset" / "task4_1" / "train.txt"
TEST_PATH = ROOT / "dataset" / "task4_1" / "test_features.txt"
OUT_PATH = ROOT / "outputs" / "task4_1_predictions.csv"
FIG_DIR = ROOT / "outputs" / "task4_1_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. парсинг и кодирование
NUC = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_file(path, has_label):
    ids, labels, seqs = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            ids.append(p[0])  # p[1] - пол, его не используем
            if has_label:
                labels.append(p[2])
                seqs.append(p[3:])
            else:
                seqs.append(p[2:])
    arr = np.array([[NUC[n] for n in s] for s in seqs], dtype=np.int8)  # (люди, 10101)
    return ids, labels, arr


tr_ids, tr_labels, tr_seq = read_file(TRAIN_PATH, True)
te_ids, _, te_seq = read_file(TEST_PATH, False)
classes = sorted(set(tr_labels))
cls2id = {c: i for i, c in enumerate(classes)}
y_lab = np.array([cls2id[c] for c in tr_labels])
n_lab, n_test = len(tr_ids), len(te_ids)
print("размечено:", n_lab, "отложено:", n_test, "позиций:", tr_seq.shape[1])
print("баланс классов:", {c: int((y_lab == i).sum()) for i, c in enumerate(classes)})

all_seq = np.vstack([tr_seq, te_seq])  # узлы 0..n_lab-1 размечены, дальше - отложенные


def one_hot(seq):
    N, L = seq.shape
    out = np.zeros((N, L, 4), dtype=np.float32)
    out[np.arange(N)[:, None], np.arange(L)[None, :], seq] = 1.0
    return out.reshape(N, L * 4)


X_oh = one_hot(all_seq)
# PCA обучается только на признаках, следовательно утечки меток нет
X = PCA(n_components=N_PCA, random_state=SEED).fit_transform(X_oh).astype(np.float32)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
del X_oh

# 2. граф kNN
A = kneighbors_graph(X, n_neighbors=K_NEIGHBORS, mode="connectivity", include_self=False)
A = ((A + A.T) > 0).astype(np.float32).tocoo()  # симметризуем: сходство взаимно
edge_index = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
print("граф:", X.shape[0], "узлов,", edge_index.shape[1] // 2, "рёбер")

# 3. маски
idx = np.arange(n_lab)
idx_trval, idx_te = train_test_split(idx, test_size=0.2, stratify=y_lab, random_state=SEED)
idx_tr, idx_val = train_test_split(idx_trval, test_size=0.25, stratify=y_lab[idx_trval], random_state=SEED)  # 60/20/20


def mask(ix):
    m = torch.zeros(X.shape[0], dtype=torch.bool)
    m[ix] = True
    return m


tr_m, val_m, te_m = mask(idx_tr), mask(idx_val), mask(idx_te)
y_full = torch.full((X.shape[0],), -1, dtype=torch.long)
y_full[:n_lab] = torch.from_numpy(y_lab)
data = Data(x=torch.from_numpy(X), edge_index=edge_index, y=y_full)

# рисунок: подграф из 100 узлов, цвет = популяция
rng = np.random.RandomState(SEED)
sub = rng.choice(n_lab, 100, replace=False)
G = nx.Graph()
G.add_nodes_from(sub.tolist())
sub_set = set(sub.tolist())
for a, b in zip(A.row, A.col):
    if a < b and a in sub_set and b in sub_set:
        G.add_edge(int(a), int(b))
palette = sns.color_palette("tab10", len(classes))
plt.figure(figsize=(8, 7))
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=SEED, k=0.4), node_color=[palette[y_lab[n]] for n in G.nodes],
                 node_size=60, with_labels=False, edge_color="#bbbbbb", width=0.6)
plt.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=palette[i], label=c) for i, c in enumerate(classes)],
           loc="best", fontsize=8)
plt.title("Подграф из 100 узлов (цвет - популяция)")
plt.axis("off"); plt.tight_layout()
plt.savefig(FIG_DIR / "subgraph_100.png", dpi=110); plt.close()


# 4. модель
class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, n_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.5, self.training)
        return self.c2(x, edge_index)


counts = np.bincount(y_lab[idx_tr], minlength=len(classes))
class_w = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32)  # редкие классы весят больше
model = GCN(N_PCA, HIDDEN, len(classes))
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)


def macro_f1(out, m):
    return f1_score(data.y[m].numpy(), out[m].argmax(1).numpy(), average="macro")


hist = {"tr_f1": [], "val_f1": []}
best_f1, best_state, bad = -1.0, None, 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    F.cross_entropy(out[tr_m], data.y[tr_m], weight=class_w).backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    tr_f1, val_f1 = macro_f1(out, tr_m), macro_f1(out, val_m)
    hist["tr_f1"].append(tr_f1); hist["val_f1"].append(val_f1)
    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch {epoch:03d} | train F1 {tr_f1:.3f} | val F1 {val_f1:.3f}")
    if val_f1 > best_f1:
        best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print("ранняя остановка на эпохе", epoch)
            break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)

# 5. оценка на тестовой маске + сравнение с обычным классификатором
y_te = data.y[te_m].numpy()
pred_te = out[te_m].argmax(1).numpy()
print(f"\nGCN  TEST accuracy = {accuracy_score(y_te, pred_te):.4f}, macro F1 = {f1_score(y_te, pred_te, average='macro'):.4f}")
print(classification_report(y_te, pred_te, target_names=classes, digits=3))

# берём узлы через ту же маску te_m, чтобы порядок строк совпал с y_te
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr_m.numpy()], data.y[tr_m].numpy())
pred_lr = lr.predict(X[te_m.numpy()])
print(f"LogReg (без графа) TEST accuracy = {accuracy_score(y_te, pred_lr):.4f}, macro F1 = {f1_score(y_te, pred_lr, average='macro'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_te, pred_te), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("предсказано"); plt.ylabel("истина"); plt.title("GCN: матрица ошибок (тестовая маска)")
plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=110); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(hist["tr_f1"], label="train macro F1"); plt.plot(hist["val_f1"], label="val macro F1")
plt.xlabel("эпоха"); plt.legend(); plt.title("GCN: кривые F1")
plt.tight_layout(); plt.savefig(FIG_DIR / "f1_curves.png", dpi=110); plt.close()

# 6. предсказания для отложенных людей
pred_new = out[n_lab:].argmax(1).numpy()
pred_df = pd.DataFrame({"sample_id": te_ids, "prediction": [classes[i] for i in pred_new]})
assert len(pred_df) == n_test and pred_df["sample_id"].is_unique and pred_df["prediction"].notna().all()
pred_df.to_csv(OUT_PATH, index=False)
print("сохранено:", OUT_PATH, pred_df.shape)
print(pred_df["prediction"].value_counts().to_dict())
