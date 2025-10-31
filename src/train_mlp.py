# src/train_mlp.py
from __future__ import annotations
from pathlib import Path
import argparse, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TEST  = DATA_DIR / "test.csv"

# ----------------------------
# Dataset
# ----------------------------
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)
        # y dtype은 task에 따라 결정(아래에서 보장)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, pdrop: float = 0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(pdrop)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # 이진: (N,1) / 멀티: (N,C)

# ----------------------------
# Utilities
# ----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_xy(path: Path, task: str):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("PC")]].values.astype(np.float32)

    y_mc = df["Attack Type"].values
    if task == "binary":
        y = (y_mc != "BENIGN").astype(np.int64)   # BENIGN=0, ATTACK=1
    else:
        # 멀티클래스: 문자열 라벨 → 인덱스
        classes = sorted(np.unique(y_mc))
        cls2id = {c:i for i,c in enumerate(classes)}
        y = np.array([cls2id[v] for v in y_mc], dtype=np.int64)
        return X, y, classes

    return X, y, None

def make_loaders(X: np.ndarray, y: np.ndarray, batch: int, seed: int, stratify=True):
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y if stratify else None
    )
    tr_ds = TabDataset(Xtr, ytr)
    va_ds = TabDataset(Xva, yva)
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False)
    va_ld = DataLoader(va_ds, batch_size=batch*2, shuffle=False, num_workers=0, pin_memory=False)
    return tr_ld, va_ld, (Xtr, Xva, ytr, yva)

def compute_pos_weight(y: np.ndarray):
    # BENIGN=0, ATTACK=1 비율로 pos_weight 계산 (neg/pos)
    pos = (y==1).sum()
    neg = (y==0).sum()
    if pos == 0: 
        return torch.tensor(1.0)
    return torch.tensor(max(neg/pos, 1.0), dtype=torch.float32)

def early_stopping_update(best, current, patience, counter):
    if current > best:
        return current, 0, True
    counter += 1
    stop = counter >= patience
    return best, counter, stop

# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, task: str):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        # 이진: (N,1) / 멀티: (N,C)
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        # Ensure target dtype for BCEWithLogitsLoss (float)
        if task == "binary":
            yb = yb.float()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, task: str):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if task == "binary":
            # 로짓 → 확률 → 0/1
            prob = torch.sigmoid(logits.squeeze(1))
            pred = (prob >= 0.5).long().cpu().numpy()
        else:
            pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a small MLP on PCA-processed CICIDS data.")
    ap.add_argument("--task", choices=["binary","multi"], default="binary", help="binary or multi-class")
    ap.add_argument("--limit", type=int, default=150_000, help="train sample upper bound (빠른 학습용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=str, default="256,128", help="예: '256,128' 또는 '512,256,128'")
    args = ap.parse_args()

    device = get_device()
    print(f"[INFO] device: {device.type}")

    # 데이터 로드
    Xtr_all, ytr_all, classes = load_xy(TRAIN, args.task)
    Xte, yte, _ = load_xy(TEST, args.task)

    # 속도 위해 학습셋 일부만 사용
    if args.limit and args.limit < len(ytr_all):
        Xtr_all, _, ytr_all, _ = train_test_split(
            Xtr_all, ytr_all, train_size=args.limit, random_state=args.seed, stratify=ytr_all
        )

    in_dim = Xtr_all.shape[1]
    print(f"[INFO] PCs: {in_dim}  train={len(ytr_all):,}  test={len(yte):,}")

    # DataLoader
    train_loader, valid_loader, _ = make_loaders(Xtr_all, ytr_all, batch=args.batch, seed=args.seed)

    # 모델/손실/옵티마이저
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    out_dim = 1 if args.task == "binary" else (len(classes))
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, pdrop=0.2).to(device)

    if args.task == "binary":
        pos_w = compute_pos_weight(ytr_all).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 학습 루프 + 얼리 스톱
    best_acc, patience_cnt = -1.0, 0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.task)
        va_acc, _, _ = evaluate(model, valid_loader, device, args.task)
        print(f"[EP {ep:02d}/{args.epochs}] train_loss={tr_loss:.4f}  val_acc={va_acc:.4f}")
        best_acc, patience_cnt, stop = early_stopping_update(best_acc, va_acc, args.patience, patience_cnt)
        if stop:
            print("[EARLY STOP] patience reached.")
            break

    train_time = time.perf_counter() - t0

    # 테스트 평가
    te_ds = TabDataset(Xte, yte)
    te_loader = DataLoader(te_ds, batch_size=args.batch*2, shuffle=False)
    te_acc, y_true, y_pred = evaluate(model, te_loader, device, args.task)

    # 리포트
    if args.task == "binary":
        print(f"\n[Test] MLP Binary acc: {te_acc:.4f}")
        print("[Report]\n", classification_report(y_true, y_pred, target_names=["BENIGN","ATTACK"], digits=4))
    else:
        target_names = classes
        print(f"\n[Test] MLP Multiclass acc: {te_acc:.4f}")
        print("[Report]\n", classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # 저장
    if args.task == "binary":
        out_pt   = MODEL_DIR / "mlp_bin.pt"
        out_meta = MODEL_DIR / "mlp_bin.meta.json"
    else:
        out_pt   = MODEL_DIR / "mlp_multi.pt"
        out_meta = MODEL_DIR / "mlp_multi.meta.json"

    torch.save({"state_dict": model.state_dict(),
                "in_dim": in_dim,
                "hidden": hidden,
                "out_dim": out_dim,
                "task": args.task,
                "classes": classes}, out_pt)

    meta = {
        "task": args.task,
        "acc_test": float(te_acc),
        "train_time_sec": round(train_time, 2),
        "in_dim": in_dim,
        "hidden": hidden,
        "out_dim": out_dim,
        "classes": classes
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] model -> {out_pt}")
    print(f"[SAVE] meta  -> {out_meta}")


if __name__ == "__main__":
    main()