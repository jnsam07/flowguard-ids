# src/train_cnn1d.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TEST  = DATA_DIR / "test.csv"


# -----------------------------
# Utility: device pick (MPS->CUDA->CPU)
# -----------------------------

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Data helpers
# -----------------------------

def load_xy(path: Path):
    df = pd.read_csv(path)
    pc_cols = [c for c in df.columns if c.startswith("PC")]  # PCA 성분들
    X = df[pc_cols].to_numpy(dtype=np.float32)
    y_mc = df["Attack Type"].astype(str)
    y_bin = (y_mc != "BENIGN").astype(np.int64).to_numpy()
    return X, y_mc.to_numpy(), y_bin, pc_cols


def to_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, target_dtype=None) -> DataLoader:
    # CNN(1D)을 위해 [N, 1, L] 형태로 바꿔줌
    X_t = torch.from_numpy(X).unsqueeze(1)  # [N, 1, L]
    y_t = torch.from_numpy(y)
    if target_dtype is not None:
        y_t = y_t.to(target_dtype)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=False)


# -----------------------------
# Model (작고 빠른 1D CNN)
# -----------------------------

class TinyCNN1D(nn.Module):
    def __init__(
        self,
        in_len: int,
        num_classes: int,
        binary: bool = False,
        p_drop: float = 0.1,
        channels: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.binary = binary
        c1, c2 = channels if channels is not None else (32, 64)
        self.hidden_channels = (int(c1), int(c2))
        self.conv1 = nn.Conv1d(1, self.hidden_channels[0], kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(self.hidden_channels[0])
        self.conv2 = nn.Conv1d(self.hidden_channels[0], self.hidden_channels[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(self.hidden_channels[1])
        self.drop  = nn.Dropout(p_drop)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(self.hidden_channels[1], 1 if binary else num_classes)

    def forward(self, x):  # x: [N,1,L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)  # [N,64]
        x = self.fc(x)
        return x  # logits


# -----------------------------
# Train / Eval routines
# -----------------------------

def train_epoch(model, loader, optimizer, criterion, device, amp: bool = True):
    use_amp = amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(xb)
                loss = criterion(logits.squeeze(), yb) if (logits.ndim == 2 and logits.size(1) == 1) else criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits.squeeze(), yb) if (logits.ndim == 2 and logits.size(1) == 1) else criterion(logits, yb)
            loss.backward()
            optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def eval_epoch_binary(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()
            ps.append(prob)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["BENIGN","ATTACK"], digits=4)
    return acc, report


def eval_epoch_multi(model, loader, device, le: LabelEncoder):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = accuracy_score(y_true, y_pred)
    target_names = list(le.classes_)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return acc, report


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train lightweight 1D-CNN on PCA features (binary/multi)")
    parser.add_argument("--task", choices=["binary", "multi"], default="binary")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--limit", type=int, default=120_000, help="훈련 샘플 상한 (층화표집)")
    parser.add_argument("--patience", type=int, default=3, help="조기종료 patience")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print(f"[INFO] device: {device}")

    # Load train/test
    Xtr, ytr_mc, ytr_bin, pc_cols = load_xy(TRAIN)
    Xte, yte_mc, yte_bin, _       = load_xy(TEST)
    in_len = Xtr.shape[1]
    print(f"[INFO] PCs: {in_len}  train={Xtr.shape[0]:,}  test={Xte.shape[0]:,}")

    # Stratified sub-sample to speed up
    if args.limit and args.limit < len(ytr_mc):
        if args.task == "binary":
            _, Xtr, _, ytr = train_test_split(np.arange(len(ytr_bin)), ytr_bin, train_size=args.limit,
                                              stratify=ytr_bin, random_state=args.seed)
        else:
            _, Xtr, _, ytr = train_test_split(np.arange(len(ytr_mc)), ytr_mc, train_size=args.limit,
                                              stratify=ytr_mc, random_state=args.seed)
        Xtr = Xtr.astype(int)
        if args.task == "binary":
            X_train = Xtr
            y_train = ytr
            X_train = load_xy(TRAIN)[0][X_train]
        else:
            X_train = load_xy(TRAIN)[0][Xtr]
            y_train = ytr
    else:
        X_train = Xtr
        y_train = ytr_bin if args.task == "binary" else ytr_mc

    # Train/Val split
    if args.task == "binary":
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1,
                                                    stratify=y_train, random_state=args.seed)
    else:
        le = LabelEncoder()
        y_idx = le.fit_transform(y_train)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_idx, test_size=0.1,
                                                    stratify=y_idx, random_state=args.seed)

    # DataLoaders
    if args.task == "binary":
        train_loader = to_loader(X_tr, y_tr.astype(np.int64), batch=args.batch, shuffle=True,  target_dtype=torch.float32)
        val_loader   = to_loader(X_val, y_val.astype(np.int64), batch=args.batch, shuffle=False, target_dtype=torch.float32)
    else:
        train_loader = to_loader(X_tr, y_tr.astype(np.int64), batch=args.batch, shuffle=True,  target_dtype=torch.long)
        val_loader   = to_loader(X_val, y_val.astype(np.int64), batch=args.batch, shuffle=False, target_dtype=torch.long)

    # Model / Loss
    cnn_channels = (32, 64)
    if args.task == "binary":
        model = TinyCNN1D(in_len=in_len, num_classes=2, binary=True, channels=cnn_channels).to(device)
        # pos_weight = (#neg / #pos) for BCEWithLogitsLoss
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        pw = torch.tensor([max(1.0, neg / max(1, pos))], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        n_class = len(np.unique(y_tr))
        model = TinyCNN1D(in_len=in_len, num_classes=n_class, binary=False, channels=cnn_channels).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------- Training loop with early stopping --------
    best_val = -1.0
    patience = args.patience
    no_improve = 0
    t0_all = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, amp=True)

        # Validation (accuracy)
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                if args.task == "binary":
                    pred = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy()
                else:
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(pred)
                ys.append(yb.numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        val_acc = accuracy_score(y_true, y_pred)
        dt = time.perf_counter() - t0
        print(f"[EP {ep:02d}/{args.epochs}] train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.2f}s)")

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            no_improve = 0
            # Save best checkpoint (temp)
            tmp_path = MODEL_DIR / ("cnn1d_bin.pt" if args.task == "binary" else "cnn1d_multi.pt")
            torch.save({
                "model_state": model.state_dict(),
                "in_len": in_len,
                "hidden_channels": list(model.hidden_channels),
            }, tmp_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EARLY STOP] patience {patience} reached.")
                break

    print(f"[DONE] train time: {time.perf_counter()-t0_all:.2f}s  best_val_acc={best_val:.4f}")

    # -------- Test evaluation --------
    if args.task == "binary":
        te_loader = to_loader(Xte, yte_bin.astype(np.int64), batch=args.batch, shuffle=False, target_dtype=torch.float32)
        acc, report = eval_epoch_binary(model, te_loader, device)
        print(f"\n[Test] Binary CNN acc: {acc:.4f}")
        print("[Report]\n" + report)
        # Save final
        out_path = MODEL_DIR / "cnn1d_bin.pt"
        meta = {
            "type": "binary",
            "algo": "TinyCNN1D",
            "val_acc": float(best_val),
            "test_acc": float(acc),
            "hidden_channels": list(model.hidden_channels),
            "pc_features": len(pc_cols),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save({
            "model_state": model.state_dict(),
            "in_len": in_len,
            "hidden_channels": list(model.hidden_channels),
        }, out_path)
        (MODEL_DIR / "cnn1d_bin.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[SAVE] {out_path}")
    else:
        # label encode test labels using same classes as training
        le = LabelEncoder().fit(y_train if isinstance(y_train, np.ndarray) else np.array(y_train))
        yte_idx = le.transform(yte_mc)
        te_loader = to_loader(Xte, yte_idx.astype(np.int64), batch=args.batch, shuffle=False, target_dtype=torch.long)
        acc, report = eval_epoch_multi(model, te_loader, device, le)
        print(f"\n[Test] Multiclass CNN acc: {acc:.4f}")
        print("[Report]\n" + report)
        out_path = MODEL_DIR / "cnn1d_multi.pt"
        meta = {
            "type": "multiclass",
            "algo": "TinyCNN1D",
            "classes": le.classes_.tolist(),
            "val_acc": float(best_val),
            "test_acc": float(acc),
            "hidden_channels": list(model.hidden_channels),
            "pc_features": len(pc_cols),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save({
            "model_state": model.state_dict(),
            "in_len": in_len,
            "hidden_channels": list(model.hidden_channels),
            "classes": le.classes_.tolist()
        }, out_path)
        (MODEL_DIR / "cnn1d_multi.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
