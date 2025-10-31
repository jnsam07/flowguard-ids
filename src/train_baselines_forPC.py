# src/train_baselines_forPC.py
from __future__ import annotations
from pathlib import Path
import argparse
import time
import json

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"

MODEL_DIR = BASE / "models" / "pc"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_xy(path: Path):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("PC")]]
    y_mc = df["Attack Type"]
    y_bin = (y_mc != "BENIGN").astype(int)
    return X, y_mc, y_bin


def stratified_limit(X, y, limit: int | None, seed: int):
    if limit is None or limit <= 0 or limit >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=limit, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def write_meta(fname: str, payload: dict):
    meta_path = MODEL_DIR / fname
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="Desktop-grade baselines on PCA CICIDS dataset.")
    ap.add_argument("--cv", type=int, default=0, help="k-fold cross validation (0이면 미실행)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Logistic / RF 공통 학습 샘플 상한 (0이면 전체 사용)")
    ap.add_argument("--svm-limit", type=int, default=200_000, help="SVM 학습 샘플 상한 (0이면 전체)")
    ap.add_argument("--dt-limit", type=int, default=150_000)
    ap.add_argument("--knn-limit", type=int, default=150_000)
    ap.add_argument("--mode", choices=["both", "binary", "multi"], default="both")
    args = ap.parse_args()

    print(f"[INFO] loading train/test from {DATA_DIR}")
    Xtr, ytr_mc, ytr_bin = load_xy(TRAIN)
    Xte, yte_mc, yte_bin = load_xy(TEST)

    limit_common = None if args.limit <= 0 else args.limit
    limit_svm = None if args.svm_limit <= 0 else args.svm_limit
    limit_dt = None if args.dt_limit <= 0 else args.dt_limit
    limit_knn = None if args.knn_limit <= 0 else args.knn_limit

    if args.mode in ("both", "binary"):
        print("\n=== Binary classification (BENIGN vs ATTACK) ===")

        # Logistic Regression (balanced, larger iteration budget)
        X_lr, y_lr = stratified_limit(Xtr, ytr_bin, limit_common, args.seed)
        print(f"[LR ] train size: {len(y_lr):,}")
        t0 = time.perf_counter()
        lr = LogisticRegression(
            max_iter=5000,
            solver="saga",
            penalty="l2",
            C=1.0,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.seed,
        )
        lr.fit(X_lr, y_lr)
        lr_pred = lr.predict(Xte)
        lr_acc = accuracy_score(yte_bin, lr_pred)
        print(f"[LR ] acc={lr_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
        dump(lr, MODEL_DIR / "lr_pc_bin.joblib")
        write_meta("lr_pc_bin.meta.json", {
            "type": "binary",
            "algo": "LogisticRegression(saga,balanced)",
            "train_samples": len(y_lr),
            "test_acc": float(lr_acc),
        })
        print("[LR ] report\n", classification_report(yte_bin, lr_pred, target_names=["BENIGN", "ATTACK"], digits=4))
        if args.cv and args.cv > 1:
            tcv = time.perf_counter()
            lr_cv = cross_val_score(
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    penalty="l2",
                    C=1.0,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
                X_lr,
                y_lr,
                cv=args.cv,
                n_jobs=-1,
            ).mean()
            print(f"[LR ] cv{args.cv}={lr_cv:.4f}  elapsed={time.perf_counter() - tcv:.2f}s")

        # Support Vector Machine (RBF kernel)
        X_svm, y_svm = stratified_limit(Xtr, ytr_bin, limit_svm, args.seed)
        print(f"[SVM] train size: {len(y_svm):,}")
        t0 = time.perf_counter()
        svm = SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=args.seed,
        )
        svm.fit(X_svm, y_svm)
        svm_pred = svm.predict(Xte)
        svm_acc = accuracy_score(yte_bin, svm_pred)
        print(f"[SVM] acc={svm_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
        dump(svm, MODEL_DIR / "svm_pc_bin.joblib")
        write_meta("svm_pc_bin.meta.json", {
            "type": "binary",
            "algo": "SVC(rbf,balanced)",
            "train_samples": len(y_svm),
            "test_acc": float(svm_acc),
        })
        print("[SVM] report\n", classification_report(yte_bin, svm_pred, target_names=["BENIGN", "ATTACK"], digits=4))
        if args.cv and args.cv > 1:
            tcv = time.perf_counter()
            svm_cv = cross_val_score(
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
                X_svm,
                y_svm,
                cv=args.cv,
                n_jobs=-1,
            ).mean()
            print(f"[SVM] cv{args.cv}={svm_cv:.4f}  elapsed={time.perf_counter() - tcv:.2f}s")

    if args.mode in ("both", "multi"):
        print("\n=== Multiclass classification (Attack Type) ===")

        # Random Forest
        X_rf, y_rf = stratified_limit(Xtr, ytr_mc, limit_common, args.seed)
        print(f"[RF ] train size: {len(y_rf):,}")
        t0 = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=args.seed,
        )
        rf.fit(X_rf, y_rf)
        rf_pred = rf.predict(Xte)
        rf_acc = accuracy_score(yte_mc, rf_pred)
        print(f"[RF ] acc={rf_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
        dump(rf, MODEL_DIR / "rf_pc_multi.joblib")
        write_meta("rf_pc_multi.meta.json", {
            "type": "multiclass",
            "algo": "RandomForestClassifier",
            "train_samples": len(y_rf),
            "test_acc": float(rf_acc),
        })
        print("[RF ] report\n", classification_report(yte_mc, rf_pred, digits=4))
        if args.cv and args.cv > 1:
            tcv = time.perf_counter()
            rf_cv = cross_val_score(rf, X_rf, y_rf, cv=args.cv, n_jobs=-1).mean()
            print(f"[RF ] cv{args.cv}={rf_cv:.4f}  elapsed={time.perf_counter() - tcv:.2f}s")

        # Decision Tree
        X_dt, y_dt = stratified_limit(Xtr, ytr_mc, limit_dt, args.seed)
        print(f"[DT ] train size: {len(y_dt):,}")
        t0 = time.perf_counter()
        dt = DecisionTreeClassifier(
            max_depth=14,
            min_samples_split=4,
            random_state=args.seed,
        )
        dt.fit(X_dt, y_dt)
        dt_pred = dt.predict(Xte)
        dt_acc = accuracy_score(yte_mc, dt_pred)
        print(f"[DT ] acc={dt_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
        dump(dt, MODEL_DIR / "dt_pc_multi.joblib")
        write_meta("dt_pc_multi.meta.json", {
            "type": "multiclass",
            "algo": "DecisionTreeClassifier",
            "train_samples": len(y_dt),
            "test_acc": float(dt_acc),
        })
        print("[DT ] report\n", classification_report(yte_mc, dt_pred, digits=4))

        # K Nearest Neighbours
        X_knn, y_knn = stratified_limit(Xtr, ytr_mc, limit_knn, args.seed)
        print(f"[KNN] train size: {len(y_knn):,}")
        t0 = time.perf_counter()
        knn = KNeighborsClassifier(n_neighbors=10, algorithm="auto", n_jobs=-1)
        knn.fit(X_knn, y_knn)
        knn_pred = knn.predict(Xte)
        knn_acc = accuracy_score(yte_mc, knn_pred)
        print(f"[KNN] acc={knn_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
        dump(knn, MODEL_DIR / "knn_pc_multi.joblib")
        write_meta("knn_pc_multi.meta.json", {
            "type": "multiclass",
            "algo": "KNeighborsClassifier",
            "train_samples": len(y_knn),
            "test_acc": float(knn_acc),
        })
        print("[KNN] report\n", classification_report(yte_mc, knn_pred, digits=4))


if __name__ == "__main__":
    main()
