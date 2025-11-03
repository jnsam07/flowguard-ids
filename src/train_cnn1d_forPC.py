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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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
        # 향상: Shortcut에도 BatchNorm 추가
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = None

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
        
        # 향상: 더 깊은 Classifier + Label Smoothing 대비
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels[-1], self.fc_hidden),
            nn.BatchNorm1d(self.fc_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_hidden, self.fc_hidden // 2),
            nn.BatchNorm1d(self.fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.fc_hidden // 2, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device, use_amp: bool, scheduler=None):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda'):
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
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_binary(model, loader, device, threshold: float = 0.5):
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
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
    return acc, precision, recall, f1, report, y_true, y_pred


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
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    return acc, precision, recall, f1, report


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float = 0.5,
):
    """검증 데이터에서 F1을 최대화하는 분류 임계값을 탐색."""
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
    parser = argparse.ArgumentParser(description="Train a deeper 1D CNN for desktop GPU environments.")
    parser.add_argument("--task", choices=["binary", "multi"], default="binary")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체 학습 데이터 사용")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.0, help="Early stopping 최소 개선 폭 (F1 기준)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=str, default="128,256,512,256", help="예: '128,256,512,256'")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--fc-hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--cv-folds", type=int, default=0, help="Cross-validation folds (0이면 미실행)")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for multi-class (0~0.2)")
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
            Xtr, y_bin_sel, test_size=0.15, stratify=y_bin_sel, random_state=args.seed
        )
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        # Precision 향상: pos_weight를 더 크게 설정하여 False Positive 감소
        pos_weight = torch.tensor(max(neg / max(pos, 1) * 1.5, 1.0), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[INFO] Binary task - pos_weight: {pos_weight.item():.2f} (pos={pos:,}, neg={neg:,})")
    else:
        le = LabelEncoder()
        y_idx = le.fit_transform(y_mc_sel)
        classes = le.classes_.tolist()
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xtr, y_idx, test_size=0.15, stratify=y_idx, random_state=args.seed
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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



    train_loader = to_loader(
        X_tr, y_tr.astype(np.float32 if args.task == "binary" else np.int64), batch=args.batch,
        shuffle=True, target_dtype=torch.float32 if args.task == "binary" else torch.long
    )
    val_loader = to_loader(
        X_val, y_val.astype(np.float32 if args.task == "binary" else np.int64), batch=args.batch,
        shuffle=False, target_dtype=torch.float32 if args.task == "binary" else torch.long
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # OneCycleLR: 학습률을 점진적으로 증가시켰다가 감소 (더 나은 수렴)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr * 3,  # 최대 학습률
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% 동안 warm-up
        anneal_strategy='cos',
        div_factor=25.0,  # 초기 lr = max_lr / 25
        final_div_factor=10000.0  # 최종 lr = max_lr / 10000
    )

    use_amp = device.type == "cuda"
    best_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    patience_ctr = 0
    t0 = time.perf_counter()
    
    # 에포크별 메트릭 저장 (그래프 그리기 위한 히스토리)
    history = {
        'epoch': [],
        'train_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'test_cv_acc': [],  # test.csv를 사용한 CV 점수
        'lr': [],
        'val_threshold': [],
    }

    # Cross-Validation 수행 (옵션)
    if args.cv_folds > 1:
        print(f"\n[INFO] Cross-Validation with {args.cv_folds} folds")
        cv_scores = []
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        
        if args.task == "binary":
            cv_data = y_bin_sel
        else:
            cv_data = y_idx
            
        for fold, (train_idx, val_idx) in enumerate(skf.split(Xtr, cv_data), 1):
            print(f"\n--- Fold {fold}/{args.cv_folds} ---")
            X_fold_tr, X_fold_val = Xtr[train_idx], Xtr[val_idx]
            y_fold_tr, y_fold_val = cv_data[train_idx], cv_data[val_idx]
            
            # Fold 모델 생성
            fold_model = DeepCNN1D(
                in_len=in_len,
                num_classes=len(np.unique(y_fold_tr)) if args.task == "multi" else 2,
                binary=(args.task == "binary"),
                channels=channels,
                kernel_size=args.kernel_size,
                fc_hidden=args.fc_hidden,
                dropout=args.dropout,
            ).to(device)
            
            fold_optimizer = torch.optim.AdamW(fold_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            fold_train_loader_temp = to_loader(
                X_fold_tr, y_fold_tr.astype(np.float32 if args.task == "binary" else np.int64),
                batch=args.batch, shuffle=True,
                target_dtype=torch.float32 if args.task == "binary" else torch.long
            )
            fold_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                fold_optimizer,
                max_lr=args.lr * 3,
                epochs=min(15, args.epochs),
                steps_per_epoch=len(fold_train_loader_temp),
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
            
            if args.task == "binary":
                pos = (y_fold_tr == 1).sum()
                neg = (y_fold_tr == 0).sum()
                pos_weight = torch.tensor(max(neg / max(pos, 1) * 1.5, 1.0), dtype=torch.float32, device=device)
                fold_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                fold_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            
            fold_train_loader = fold_train_loader_temp
            fold_val_loader = to_loader(
                X_fold_val, y_fold_val.astype(np.float32 if args.task == "binary" else np.int64),
                batch=args.batch, shuffle=False,
                target_dtype=torch.float32 if args.task == "binary" else torch.long
            )
            
            # Fold 학습 (간단 버전)
            fold_best_acc = -1.0
            for ep in range(1, min(15, args.epochs) + 1):  # CV는 짧게
                tr_loss = train_epoch(
                    fold_model,
                    fold_train_loader,
                    fold_optimizer,
                    fold_criterion,
                    device,
                    use_amp,
                    scheduler=fold_scheduler,
                )
                fold_model.eval()
                with torch.no_grad():
                    preds, trues = [], []
                    for xb, yb in fold_val_loader:
                        xb = xb.to(device)
                        logits = fold_model(xb)
                        if args.task == "binary":
                            pred = torch.sigmoid(logits.squeeze(1)).ge(0.5).long().cpu().numpy()
                        else:
                            pred = torch.argmax(logits, dim=1).cpu().numpy()
                        preds.append(pred)
                        trues.append(yb.numpy())
                y_val_pred = np.concatenate(preds)
                y_val_true = np.concatenate(trues)
                val_acc = accuracy_score(y_val_true, y_val_pred)
                
                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc
            
            cv_scores.append(fold_best_acc)
            print(f"Fold {fold} best val_acc: {fold_best_acc:.4f}")
            
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"\n[CV Result] Mean Acc: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"[CV Scores] {[f'{s:.4f}' for s in cv_scores]}\n")

    print("[INFO] Training final model on full training set...")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_amp, scheduler=scheduler)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_probs, preds, trues = [], [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                if args.task == "binary":
                    prob = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
                    val_probs.append(prob)
                else:
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.append(pred)
                trues.append(yb.numpy())
        if args.task == "binary":
            y_val_true = np.concatenate(trues).astype(int)
            y_val_prob = np.concatenate(val_probs)
            best_info = find_best_threshold(y_val_true, y_val_prob, default_threshold=best_threshold)
            val_threshold = best_info["threshold"]
            y_val_pred = (y_val_prob >= val_threshold).astype(int)
            val_acc = accuracy_score(y_val_true, y_val_pred)
            val_prec = best_info["precision"]
            val_rec = best_info["recall"]
            val_f1 = best_info["f1"]
        else:
            y_val_pred = np.concatenate(preds)
            y_val_true = np.concatenate(trues)
            val_acc = accuracy_score(y_val_true, y_val_pred)
            val_prec = precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
            val_rec = recall_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
            val_threshold = None
        
        # test.csv를 사용한 Cross-Validation (매 에포크마다 또는 주기적으로)
        test_cv_acc = None
        if ep % 5 == 0 or ep == args.epochs or ep == 1:  # 1, 5, 10, ... 에포크 및 마지막
            print(f"  → Running CV on test.csv at epoch {ep}...")
            cv_scores_test = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            
            if args.task == "binary":
                cv_y = yte_bin
            else:
                le_temp = LabelEncoder().fit(y_mc_sel)
                cv_y = le_temp.transform(yte_mc)
            
            for fold_idx, (_, val_idx) in enumerate(skf.split(Xte, cv_y)):
                X_fold_val = Xte[val_idx]
                y_fold_val = cv_y[val_idx]
                
                fold_val_loader = to_loader(
                    X_fold_val,
                    y_fold_val.astype(np.float32 if args.task == "binary" else np.int64),
                    batch=args.batch,
                    shuffle=False,
                    target_dtype=torch.float32 if args.task == "binary" else torch.long
                )
                
                with torch.no_grad():
                    preds_fold = []
                    for xb, yb in fold_val_loader:
                        xb = xb.to(device)
                        logits = model(xb)
                        if args.task == "binary":
                            prob = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
                            pred = (prob >= val_threshold).astype(int)
                        else:
                            pred = torch.argmax(logits, dim=1).cpu().numpy()
                        preds_fold.append(pred)
                y_fold_pred = np.concatenate(preds_fold)
                fold_acc = accuracy_score(y_fold_val, y_fold_pred)
                cv_scores_test.append(fold_acc)
            
            test_cv_acc = float(np.mean(cv_scores_test))
            print(f"  → Test CV Acc: {test_cv_acc:.4f} (±{np.std(cv_scores_test):.4f})")
        
        # 히스토리 저장
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(ep)
        history['train_loss'].append(float(tr_loss))
        history['val_acc'].append(float(val_acc))
        history['val_precision'].append(float(val_prec))
        history['val_recall'].append(float(val_rec))
        history['val_f1'].append(float(val_f1))
        history['test_cv_acc'].append(test_cv_acc)  # None인 경우도 저장
        history['lr'].append(float(current_lr))
        history['val_threshold'].append(float(val_threshold) if val_threshold is not None else None)

        extra = f" | thr={val_threshold:.3f}" if args.task == "binary" else ""
        print(f"[EP {ep:02d}/{args.epochs}] loss={tr_loss:.4f} | acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f} | lr={current_lr:.6f}{extra}")

        # Best model 저장 (F1-Score 기준으로 변경)
        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1
            best_state = model.state_dict()
            if args.task == "binary":
                best_threshold = val_threshold
            patience_ctr = 0
            print(f"  ✓ New best F1-Score: {best_f1:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[EARLY STOP] patience {args.patience} reached at epoch {ep}. Best F1: {best_f1:.4f}")
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    if args.task == "binary":
        te_loader = to_loader(
            Xte, yte_bin.astype(np.float32), batch=args.batch, shuffle=False, target_dtype=torch.float32
        )
        test_acc, test_prec, test_rec, test_f1, report, _, _ = evaluate_binary(
            model, te_loader, device, threshold=best_threshold
        )
        print(f"\n[Test] Binary CNN Metrics:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")
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
            "val_best_acc": float(best_f1),
            "val_best_f1": float(best_f1),
            "best_threshold": float(best_threshold),
            "test_acc": float(test_acc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
            "train_time_sec": round(train_time, 2),
            "pc_features": len(pc_cols),
            "history": history,  # 그래프 그리기 위한 에포크별 메트릭
        }
        if args.cv_folds > 1:
            meta["cv_mean_acc"] = float(cv_mean)
            meta["cv_std_acc"] = float(cv_std)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[SAVE] {out_path}")
    else:
        le = LabelEncoder().fit(y_mc_sel)
        yte_idx = le.transform(yte_mc)
        te_loader = to_loader(
            Xte, yte_idx.astype(np.int64), batch=args.batch, shuffle=False, target_dtype=torch.long
        )
        test_acc, test_prec, test_rec, test_f1, report = evaluate_multi(model, te_loader, device, label_names=classes)
        print(f"\n[Test] Multiclass CNN Metrics:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f} (weighted)")
        print(f"  Recall:    {test_rec:.4f} (weighted)")
        print(f"  F1-Score:  {test_f1:.4f} (weighted)")
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
            "val_best_acc": float(best_f1),
            "val_best_f1": float(best_f1),
            "test_acc": float(test_acc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
            "train_time_sec": round(train_time, 2),
            "pc_features": len(pc_cols),
            "classes": classes,
            "history": history,  # 그래프 그리기 위한 에포크별 메트릭
        }
        if args.cv_folds > 1:
            meta["cv_mean_acc"] = float(cv_mean)
            meta["cv_std_acc"] = float(cv_std)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
