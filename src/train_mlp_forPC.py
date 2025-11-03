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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


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


def standardize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std_safe = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std_safe


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
def evaluate(model, loader, device, task: str, threshold: float = 0.5, return_probs: bool = False):
    model.eval()
    preds, trues = [], []
    probs = [] if task == "binary" and return_probs else None
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if task == "binary":
            prob = torch.sigmoid(logits.squeeze(1))
            if probs is not None:
                probs.append(prob.cpu().numpy())
            pred = prob.ge(threshold).long().cpu().numpy()
        else:
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    # 모든 메트릭 계산
    if task == "binary":
        prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    else:
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    y_prob = np.concatenate(probs) if probs is not None else None
    return acc, prec, rec, f1, y_true, y_pred, y_prob


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float = 0.5,
):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)

    best = {
        "threshold": float(default_threshold),
        "precision": precision_score(y_true, y_prob >= default_threshold, zero_division=0),
        "recall": recall_score(y_true, y_prob >= default_threshold, zero_division=0),
        "f1": f1_score(y_true, y_prob >= default_threshold, zero_division=0),
    }

    if thresholds.size > 0:
        f1_scores = 2 * precision_vals[:-1] * recall_vals[:-1] / (
            precision_vals[:-1] + recall_vals[:-1] + 1e-12
        )
        f1_scores = np.nan_to_num(f1_scores, nan=0.0)
        best_idx = int(np.argmax(f1_scores))
        candidate = {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision_vals[:-1][best_idx]),
            "recall": float(recall_vals[:-1][best_idx]),
            "f1": float(f1_scores[best_idx]),
        }
        if candidate["f1"] > best["f1"]:
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description="Train a large MLP for desktop hardware.")
    parser.add_argument("--task", choices=["binary", "multi"], default="binary")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체 학습 데이터 사용")
    parser.add_argument("--hidden", type=str, default="2048,1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--min-delta", type=float, default=0.0, help="조기 종료 시 요구되는 최소 F1 향상 폭")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="멀티클래스 CrossEntropy 라벨 스무딩")
    parser.add_argument("--pos-weight-scale", type=float, default=1.5, help="Binary pos_weight 배율")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 이하이면 그래디언트 클리핑 비활성화")
    parser.add_argument("--no-standardize", action="store_false", dest="standardize", help="PC 특성 표준화를 비활성화")
    parser.set_defaults(standardize=True)
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
    feature_mean = None
    feature_std = None
    if args.standardize:
        feature_mean = Xtr.mean(axis=0)
        feature_std = Xtr.std(axis=0)
        Xtr = standardize_features(Xtr, feature_mean, feature_std)
        Xte = standardize_features(Xte, feature_mean, feature_std)
    in_dim = Xtr.shape[1]
    print(f"[INFO] PCs: {in_dim}  train={len(ytr):,}  test={len(yte):,}")

    train_loader, valid_loader, _ = make_loaders(Xtr, ytr, args.batch, args.seed)

    hidden = [int(v.strip()) for v in args.hidden.split(",") if v.strip()]
    out_dim = 1 if args.task == "binary" else len(classes)
    model = LargeMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=args.dropout).to(device)

    pos_weight_value = None
    if args.task == "binary":
        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        ratio = neg / max(pos, 1)
        pos_weight_value = max(ratio * args.pos_weight_scale, 1.0)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # OneCycleLR로 변경 (CNN과 동일)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 3,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
    )
    
    # Mixed precision 사용 (RTX 4070 Super 최적화)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    best_f1 = -1.0
    best_threshold = 0.5
    patience_ctr = 0
    best_state = None
    t0 = time.perf_counter()
    
    # 에포크별 메트릭 저장
    history = {
        'train_loss': [],
        'val_acc': [],
        'val_prec': [],
        'val_rec': [],
        'val_f1': [],
        'cv_acc': [],
        'lr': [],
        'val_threshold': [],
    }

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(xb)
                if args.task == "binary":
                    loss = criterion(logits.squeeze(1), yb.float())
                else:
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * xb.size(0)

        tr_loss = total / len(train_loader.dataset)
        history['train_loss'].append(tr_loss)

        val_threshold = None
        val_acc, val_prec, val_rec, val_f1, y_val_true, y_val_pred, val_prob = evaluate(
            model,
            valid_loader,
            device,
            args.task,
            threshold=best_threshold,
            return_probs=args.task == 'binary',
        )
        if args.task == 'binary' and val_prob is not None:
            best_info = find_best_threshold(y_val_true, val_prob, default_threshold=best_threshold)
            val_threshold = best_info['threshold']
            y_val_pred = (val_prob >= val_threshold).astype(int)
            val_acc = accuracy_score(y_val_true, y_val_pred)
            val_prec = best_info['precision']
            val_rec = best_info['recall']
            val_f1 = best_info['f1']
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        history['val_threshold'].append(float(val_threshold) if val_threshold is not None else None)

        # Cross-validation using test set (????? CV ??)
        cv_mean = None
        if args.cv_folds > 1 and (ep % 5 == 0 or ep == args.epochs):
            cv_scores = []
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
            cv_threshold = val_threshold if val_threshold is not None else best_threshold
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(Xte, yte)):
                X_fold_tr, X_fold_va = Xte[tr_idx], Xte[va_idx]
                y_fold_tr, y_fold_va = yte[tr_idx], yte[va_idx]

                fold_tr_ds = TensorDataset(
                    torch.tensor(X_fold_tr, dtype=torch.float32),
                    torch.tensor(y_fold_tr),
                )
                fold_va_ds = TensorDataset(
                    torch.tensor(X_fold_va, dtype=torch.float32),
                    torch.tensor(y_fold_va),
                )
                fold_tr_ld = DataLoader(fold_tr_ds, batch_size=args.batch, shuffle=False)
                fold_va_ld = DataLoader(fold_va_ds, batch_size=args.batch * 2, shuffle=False)

                fold_acc, _, _, _, _, _, _ = evaluate(
                    model, fold_va_ld, device, args.task, threshold=cv_threshold
                )
                cv_scores.append(fold_acc)

            cv_mean = float(np.mean(cv_scores))
        history['cv_acc'].append(cv_mean)
        lr_curr = optimizer.param_groups[0]['lr']
        thr_info = ''
        if args.task == 'binary' and val_threshold is not None:
            thr_info = f' | thr={val_threshold:.3f}'
        if cv_mean is not None:
            print(f"[EP {ep:02d}/{args.epochs}] loss={tr_loss:.4f} | val: acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f} | CV_acc={cv_mean:.4f} | lr={lr_curr:.5f}{thr_info}")
        else:
            print(f"[EP {ep:02d}/{args.epochs}] loss={tr_loss:.4f} | val: acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f} | lr={lr_curr:.5f}{thr_info}")

        # F1-score ???? best model ??
        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1
            best_state = model.state_dict()
            if args.task == 'binary' and val_threshold is not None:
                best_threshold = val_threshold
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[EARLY STOP] patience reached at epoch {ep}. Best F1: {best_f1:.4f}")
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
    test_acc, test_prec, test_rec, test_f1, y_true, y_pred, _ = evaluate(
        model,
        te_loader,
        device,
        args.task,
        threshold=best_threshold,
        return_probs=args.task == 'binary',
    )

    if args.task == "binary":
        print(f"\n[Test] MLP Binary - acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")
        print("[Report]\n", classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"], digits=4))
        model_path = MODEL_DIR / "mlp_pc_bin.pt"
        meta_path = MODEL_DIR / "mlp_pc_bin.meta.json"
    else:
        print(f"\n[Test] MLP Multiclass - acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}")
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
        "standardize": args.standardize,
        "feature_mean": feature_mean.tolist() if feature_mean is not None else None,
        "feature_std": feature_std.tolist() if feature_std is not None else None,
        "best_threshold": float(best_threshold) if args.task == 'binary' else None,
    }, model_path)

    meta = {
        'task': args.task,
        'hidden': hidden,
        'dropout': args.dropout,
        'train_time_sec': round(train_time, 2),
        'val_best_f1': float(best_f1),
        'test_acc': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'pc_features': in_dim,
        'standardize': args.standardize,
        'feature_mean': feature_mean.tolist() if feature_mean is not None else None,
        'feature_std': feature_std.tolist() if feature_std is not None else None,
        'best_threshold': float(best_threshold) if args.task == 'binary' else None,
        'pos_weight_scale': args.pos_weight_scale,
        'pos_weight_value': float(pos_weight_value) if args.task == 'binary' else None,
        'label_smoothing': args.label_smoothing,
        'grad_clip': args.grad_clip,
        'min_delta': args.min_delta,
        'cv_folds': args.cv_folds,
        'history': history,
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[SAVE] {model_path}")
    print(f"[SAVE] {meta_path}")



if __name__ == "__main__":
    main()
