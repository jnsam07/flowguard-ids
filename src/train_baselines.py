
# src/train_baselines.py
from __future__ import annotations
from pathlib import Path
import argparse
import time

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import json


BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "data" / "processed" / "train.csv"
TEST  = BASE / "data" / "processed" / "test.csv"

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_xy(path: Path):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("PC")]]
    y_mc = df["Attack Type"]                 # 다중분류용
    y_bin = (y_mc != "BENIGN").astype(int)   # 이진분류용 (BENIGN=0, ATTACK=1)
    return X, y_mc, y_bin


def stratified_subsample(X, y, limit: int | None, seed: int):
    """라벨 비율을 유지하며 최대 limit개만 샘플링. limit가 None이거나 더 크면 원본 반환."""
    if limit is None or limit >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=limit, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def main():
    ap = argparse.ArgumentParser(description="Fast baselines on PCA-processed CICIDS (speed-optimized).")
    ap.add_argument("--cv", type=int, default=0, help="k-fold CV (0이면 생략, 기본 0)")
    ap.add_argument("--seed", type=int, default=0, help="random_state")
    ap.add_argument("--limit", type=int, default=120_000, help="훈련 샘플 상한 (LR/RF 등에 공통 적용)")
    ap.add_argument("--svm-limit", type=int, default=60_000, help="SVM 전용 훈련 샘플 상한")
    ap.add_argument("--mode", choices=["both", "binary", "multi"], default="both",
                    help="학습 모드 선택 (both/binary/multi)")
    args = ap.parse_args()

    print(f"[INFO] load -> {TRAIN}")
    Xtr, ytr_mc, ytr_bin = load_xy(TRAIN)
    print(f"[INFO] load -> {TEST}")
    Xte, yte_mc, yte_bin = load_xy(TEST)
    print(f"[INFO] models will be saved under -> {MODEL_DIR}")

    # ===== 이진분류: LR / SVM =====
    if args.mode in ("both", "binary"):
        print("\n=== Binary: BENIGN vs ATTACK ===")
        # 공통 한도 샘플링 (LR)
        Xtr_lr, ytr_lr = stratified_subsample(Xtr, ytr_bin, args.limit, args.seed)
        print(f"[INFO] Train size (LR): {len(ytr_lr):,}")

        t0 = time.perf_counter()
        lr = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1, random_state=args.seed)
        lr.fit(Xtr_lr, ytr_lr)
        lr_pred = lr.predict(Xte)
        lr_acc = accuracy_score(yte_bin, lr_pred)
        elapsed = time.perf_counter() - t0

        print(f"[LR] acc : {lr_acc:.4f}  (fit+predict {elapsed:.2f}s)")
        # --- save LR model (binary) ---
        lr_path = MODEL_DIR / "lr_bin.joblib"
        dump(lr, lr_path)
        dump({"type": "binary", "algo": "LogisticRegression", "acc": float(lr_acc)}, MODEL_DIR / "lr_bin.meta.joblib")
        print(f"[SAVE] LR (binary) -> {lr_path}")
        if args.cv and args.cv > 1:
            tcv = time.perf_counter()
            lr_cv = cross_val_score(lr, Xtr_lr, ytr_lr, cv=args.cv, n_jobs=-1).mean()
            print(f"[LR] cv{args.cv}: {lr_cv:.4f}  ({time.perf_counter()-tcv:.2f}s)")
        print("[LR] report\n", classification_report(yte_bin, lr_pred, target_names=["BENIGN","ATTACK"], digits=4))

        # SVM은 더 작은 한도로 별도 샘플링
        Xtr_svm, ytr_svm = stratified_subsample(Xtr, ytr_bin, args.svm_limit, args.seed)
        print(f"[INFO] Train size (SVM): {len(ytr_svm):,}")

        t0 = time.perf_counter()
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=args.seed)
        svm.fit(Xtr_svm, ytr_svm)
        svm_pred = svm.predict(Xte)
        svm_acc = accuracy_score(yte_bin, svm_pred)
        elapsed = time.perf_counter() - t0

        print(f"[SVM] acc: {svm_acc:.4f}  (fit+predict {elapsed:.2f}s)")
        # --- save SVM model (binary) ---
        svm_path = MODEL_DIR / "svm_bin.joblib"
        dump(svm, svm_path)
        dump({"type": "binary", "algo": "SVC(rbf)", "acc": float(svm_acc)}, MODEL_DIR / "svm_bin.meta.joblib")
        print(f"[SAVE] SVM (binary) -> {svm_path}")
        if args.cv and args.cv > 1:
            tcv = time.perf_counter()
            svm_cv = cross_val_score(svm, Xtr_svm, ytr_svm, cv=args.cv, n_jobs=-1).mean()
            print(f"[SVM] cv{args.cv}: {svm_cv:.4f}  ({time.perf_counter()-tcv:.2f}s)")
        print("[SVM] report\n", classification_report(yte_bin, svm_pred, target_names=["BENIGN","ATTACK"], digits=4))

    # ===== 다중분류: RandomForest =====
    if args.mode in ("both", "multi"):
        print("\n=== Multiclass: Attack Type ===")
        # 공통 한도 샘플링 (RF)
        Xtr_rf, ytr_rf = stratified_subsample(Xtr, ytr_mc, args.limit, args.seed)
        print(f"[INFO] Train size (RF): {len(ytr_rf):,}")

        t0 = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features="sqrt", n_jobs=-1, random_state=args.seed
        )
        rf.fit(Xtr_rf, ytr_rf)
        rf_pred = rf.predict(Xte)
        rf_acc = accuracy_score(yte_mc, rf_pred)
        elapsed = time.perf_counter() - t0

        print(f"[RF ] acc : {rf_acc:.4f}  (fit+predict {elapsed:.2f}s)")
        # --- save RF model (multiclass) ---
        rf_path = MODEL_DIR / "rf_multi.joblib"
        dump(rf, rf_path)
        dump({"type": "multiclass", "algo": "RandomForestClassifier", "acc": float(rf_acc)}, MODEL_DIR / "rf_multi.meta.joblib")
        print(f"[SAVE] RF (multiclass) -> {rf_path}")
        if args.cv and args.cv > 1:
            tcv = time.perf_counter()
            rf_cv = cross_val_score(rf, Xtr_rf, ytr_rf, cv=args.cv, n_jobs=-1).mean()
            print(f"[RF ] cv{args.cv}: {rf_cv:.4f}  ({time.perf_counter()-tcv:.2f}s)")
        print("[RF ] report\n", classification_report(yte_mc, rf_pred, digits=4))


if __name__ == "__main__":
    main()