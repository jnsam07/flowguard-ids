# src/train_mlp_forPC.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models" / "pc"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_xy(path: Path, task: str):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("PC")]].values.astype(np.float32)
    y_mc = df["Attack Type"].values
    if task == "binary":
        y = (y_mc != "BENIGN").astype(np.int64)
        return X, y, None
    classes = sorted(np.unique(y_mc))
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2id[v] for v in y_mc], dtype=np.int64)
    return X, y, classes


def make_loaders(X: np.ndarray, y: np.ndarray, batch: int, seed: int, num_workers: int = 4):
    strat = y if len(np.unique(y)) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=strat
    )
    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr),
    )
    va_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val),
    )
    tr_ld = DataLoader(
        tr_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=batch * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return tr_ld, va_ld, (X_tr, X_val, y_tr, y_val)


class LargeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.extend([
                nn.Linear(last, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.dropout = dropout

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device, task: str):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if task == "binary":
            prob = torch.sigmoid(logits.squeeze(1))
            pred = prob.ge(0.5).long().cpu().numpy()
        else:
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train a large MLP for desktop hardware.")
    parser.add_argument("--task", choices=["binary", "multi"], default="binary")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체 학습 데이터 사용")
    parser.add_argument("--hidden", type=str, default="1024,512,256,128")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"[INFO] device: {device}")

    Xtr, ytr, classes = load_xy(TRAIN, args.task)
    Xte, yte, _ = load_xy(TEST, args.task)
    if args.limit > 0 and args.limit < len(ytr):
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr, train_size=args.limit, random_state=args.seed, stratify=ytr
        )
    in_dim = Xtr.shape[1]
    print(f"[INFO] PCs: {in_dim}  train={len(ytr):,}  test={len(yte):,}")

    train_loader, valid_loader, _ = make_loaders(Xtr, ytr, args.batch, args.seed)

    hidden = [int(v.strip()) for v in args.hidden.split(",") if v.strip()]
    out_dim = 1 if args.task == "binary" else len(classes)
    model = LargeMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=args.dropout).to(device)

    if args.task == "binary":
        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        pos_weight = torch.tensor(max(neg / max(pos, 1), 1.0), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = -1.0
    patience_ctr = 0
    best_state = None
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(xb)
                if args.task == "binary":
                    loss = criterion(logits.squeeze(1), yb.float())
                else:
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * xb.size(0)
        tr_loss = total / len(train_loader.dataset)

        val_acc, _, _ = evaluate(model, valid_loader, device, args.task)
        scheduler.step(val_acc)
        print(f"[EP {ep:02d}/{args.epochs}] train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.5f}")

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("[EARLY STOP] patience reached.")
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    te_ds = TensorDataset(
        torch.tensor(Xte, dtype=torch.float32),
        torch.tensor(yte),
    )
    te_loader = DataLoader(
        te_ds,
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_acc, y_true, y_pred = evaluate(model, te_loader, device, args.task)

    if args.task == "binary":
        print(f"\n[Test] MLP Binary acc: {test_acc:.4f}")
        print("[Report]\n", classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"], digits=4))
        model_path = MODEL_DIR / "mlp_pc_bin.pt"
        meta_path = MODEL_DIR / "mlp_pc_bin.meta.json"
    else:
        print(f"\n[Test] MLP Multiclass acc: {test_acc:.4f}")
        print("[Report]\n", classification_report(y_true, y_pred, target_names=classes, digits=4))
        model_path = MODEL_DIR / "mlp_pc_multi.pt"
        meta_path = MODEL_DIR / "mlp_pc_multi.meta.json"

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden": hidden,
        "out_dim": out_dim,
        "dropout": args.dropout,
        "task": args.task,
        "classes": classes,
    }, model_path)

    meta = {
        "task": args.task,
        "hidden": hidden,
        "dropout": args.dropout,
        "train_time_sec": round(train_time, 2),
        "val_best_acc": float(best_acc),
        "test_acc": float(test_acc),
        "pc_features": in_dim,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[SAVE] {model_path}")
    print(f"[SAVE] {meta_path}")


if __name__ == "__main__":
    main()
