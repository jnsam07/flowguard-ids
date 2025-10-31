# src/train_cnn1d_forPC.py
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models" / "pc"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_xy(path: Path):
    df = pd.read_csv(path)
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    X = df[pc_cols].to_numpy(dtype=np.float32)
    y_mc = df["Attack Type"].astype(str)
    y_bin = (y_mc != "BENIGN").astype(np.int64).to_numpy()
    return X, y_mc.to_numpy(), y_bin, pc_cols


def to_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, target_dtype=None, num_workers: int = 4):
    X_t = torch.from_numpy(X).unsqueeze(1)
    y_t = torch.from_numpy(y)
    if target_dtype is not None:
        y_t = y_t.to(target_dtype)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class DeepCNN1D(nn.Module):
    def __init__(
        self,
        in_len: int,
        num_classes: int,
        binary: bool,
        channels: tuple[int, ...],
        kernel_size: int,
        fc_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        self.fc_hidden = int(fc_hidden)
        self.dropout = float(dropout)
        self.binary = binary

        blocks = []
        in_ch = 1
        for ch in self.channels:
            blocks.append(ResBlock(in_ch, ch, self.kernel_size, self.dropout))
            in_ch = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        out_dim = 1 if binary else num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels[-1], self.fc_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_hidden, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device, use_amp: bool):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                if logits.ndim == 2 and logits.size(1) == 1:
                    loss = criterion(logits.squeeze(1), yb)
                else:
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            if logits.ndim == 2 and logits.size(1) == 1:
                loss = criterion(logits.squeeze(1), yb)
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_binary(model, loader, device):
    model.eval()
    probs, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(1)
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.append(prob)
        trues.append(yb.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
    return acc, report, y_true, y_pred


@torch.no_grad()
def evaluate_multi(model, loader, device, label_names: list[str]):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    return acc, report


def main():
    parser = argparse.ArgumentParser(description="Train a deeper 1D CNN for desktop GPU environments.")
    parser.add_argument("--task", choices=["binary", "multi"], default="binary")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체 학습 데이터 사용")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=str, default="64,128,256", help="예: '64,128,256'")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--fc-hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"[INFO] device: {device}")

    Xtr, ytr_mc, ytr_bin, pc_cols = load_xy(TRAIN)
    Xte, yte_mc, yte_bin, _ = load_xy(TEST)
    in_len = Xtr.shape[1]
    print(f"[INFO] PCs: {in_len}  train={Xtr.shape[0]:,}  test={Xte.shape[0]:,}")

    if args.limit > 0 and args.limit < len(ytr_bin):
        if args.task == "binary":
            _, sel_idx, _, _ = train_test_split(
                np.arange(len(ytr_bin)),
                ytr_bin,
                train_size=args.limit,
                stratify=ytr_bin,
                random_state=args.seed,
            )
            Xtr = Xtr[sel_idx]
            y_bin_sel = ytr_bin[sel_idx]
            y_mc_sel = ytr_mc[sel_idx]
        else:
            _, sel_idx, _, _ = train_test_split(
                np.arange(len(ytr_mc)),
                ytr_mc,
                train_size=args.limit,
                stratify=ytr_mc,
                random_state=args.seed,
            )
            Xtr = Xtr[sel_idx]
            y_mc_sel = ytr_mc[sel_idx]
            y_bin_sel = ytr_bin[sel_idx]
    else:
        y_bin_sel = ytr_bin
        y_mc_sel = ytr_mc

    classes = None
    if args.task == "binary":
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xtr, y_bin_sel, test_size=0.1, stratify=y_bin_sel, random_state=args.seed
        )
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        pos_weight = torch.tensor(max(neg / max(pos, 1), 1.0), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        le = LabelEncoder()
        y_idx = le.fit_transform(y_mc_sel)
        classes = le.classes_.tolist()
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xtr, y_idx, test_size=0.1, stratify=y_idx, random_state=args.seed
        )
        criterion = nn.CrossEntropyLoss()

    channels = tuple(int(c.strip()) for c in args.channels.split(",") if c.strip())
    model = DeepCNN1D(
        in_len=in_len,
        num_classes=len(np.unique(y_tr)) if args.task == "multi" else 2,
        binary=(args.task == "binary"),
        channels=channels,
        kernel_size=args.kernel_size,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    train_loader = to_loader(
        X_tr, y_tr.astype(np.float32 if args.task == "binary" else np.int64), batch=args.batch,
        shuffle=True, target_dtype=torch.float32 if args.task == "binary" else torch.long
    )
    val_loader = to_loader(
        X_val, y_val.astype(np.float32 if args.task == "binary" else np.int64), batch=args.batch,
        shuffle=False, target_dtype=torch.float32 if args.task == "binary" else torch.long
    )

    use_amp = device.type == "cuda"
    best_acc = -1.0
    best_state = None
    patience_ctr = 0
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_amp)
        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                if args.task == "binary":
                    pred = torch.sigmoid(logits.squeeze(1)).ge(0.5).long().cpu().numpy()
                else:
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        y_val_pred = np.concatenate(preds)
        y_val_true = np.concatenate(trues)
        val_acc = accuracy_score(y_val_true, y_val_pred)
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

    if args.task == "binary":
        te_loader = to_loader(
            Xte, yte_bin.astype(np.float32), batch=args.batch, shuffle=False, target_dtype=torch.float32
        )
        test_acc, report, _, _ = evaluate_binary(model, te_loader, device)
        print(f"\n[Test] Binary CNN acc: {test_acc:.4f}")
        print("[Report]\n" + report)
        out_path = MODEL_DIR / "cnn1d_pc_bin.pt"
        meta_path = MODEL_DIR / "cnn1d_pc_bin.meta.json"
        torch.save({
            "model_state": model.state_dict(),
            "in_len": in_len,
            "channels": list(model.channels),
            "kernel_size": model.kernel_size,
            "fc_hidden": model.fc_hidden,
            "dropout": model.dropout,
            "task": "binary",
        }, out_path)
        meta = {
            "task": "binary",
            "channels": list(model.channels),
            "kernel_size": model.kernel_size,
            "fc_hidden": model.fc_hidden,
            "dropout": model.dropout,
            "val_best_acc": float(best_acc),
            "test_acc": float(test_acc),
            "train_time_sec": round(train_time, 2),
            "pc_features": len(pc_cols),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[SAVE] {out_path}")
    else:
        le = LabelEncoder().fit(y_mc_sel)
        yte_idx = le.transform(yte_mc)
        te_loader = to_loader(
            Xte, yte_idx.astype(np.int64), batch=args.batch, shuffle=False, target_dtype=torch.long
        )
        test_acc, report = evaluate_multi(model, te_loader, device, label_names=classes)
        print(f"\n[Test] Multiclass CNN acc: {test_acc:.4f}")
        print("[Report]\n" + report)
        out_path = MODEL_DIR / "cnn1d_pc_multi.pt"
        meta_path = MODEL_DIR / "cnn1d_pc_multi.meta.json"
        torch.save({
            "model_state": model.state_dict(),
            "in_len": in_len,
            "channels": list(model.channels),
            "kernel_size": model.kernel_size,
            "fc_hidden": model.fc_hidden,
            "dropout": model.dropout,
            "classes": classes,
            "task": "multi",
        }, out_path)
        meta = {
            "task": "multi",
            "channels": list(model.channels),
            "kernel_size": model.kernel_size,
            "fc_hidden": model.fc_hidden,
            "dropout": model.dropout,
            "val_best_acc": float(best_acc),
            "test_acc": float(test_acc),
            "train_time_sec": round(train_time, 2),
            "pc_features": len(pc_cols),
            "classes": classes,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
