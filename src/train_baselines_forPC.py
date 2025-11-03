# src/train_baselines_forPC.py
from __future__ import annotations
from pathlib import Path
import os
import argparse
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, parallel_backend


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
    ap.add_argument("--limit", type=int, default=250_000, help="Logistic / RF 공통 학습 샘플 상한 (0이면 전체 사용)")
    ap.add_argument("--svm-limit", type=int, default=120_000, help="SVM 학습 샘플 상한 (0이면 전체)")
    ap.add_argument("--dt-limit", type=int, default=120_000)
    ap.add_argument("--knn-limit", type=int, default=80_000)
    ap.add_argument("--mode", choices=["both", "binary", "multi"], default="both")
    ap.add_argument("--n-jobs", type=int, default=-1, help="병렬 처리에 사용할 스레드 개수 (기본 -1: 모든 코어)")
    ap.add_argument("--omp-threads", type=int, default=0, help="라이브러리 내부 OMP 스레드 개수 (0이면 자동)")
    ap.add_argument("--model-parallel", type=int, default=max(1, (mp.cpu_count() or 4) // 2),
                    help="ThreadPoolExecutor로 동시에 학습할 모델 개수")
    args = ap.parse_args()

    print(f"[INFO] loading train/test from {DATA_DIR}")
    Xtr, ytr_mc, ytr_bin = load_xy(TRAIN)
    Xte, yte_mc, yte_bin = load_xy(TEST)

    limit_common = None if args.limit <= 0 else args.limit
    limit_svm = None if args.svm_limit <= 0 else args.svm_limit
    limit_dt = None if args.dt_limit <= 0 else args.dt_limit
    limit_knn = None if args.knn_limit <= 0 else args.knn_limit

    total_cpu = mp.cpu_count() or 1
    # 라이젠 5900X (12코어 24스레드) 최대 활용 설정
    parallel_workers = 1  # 한 번에 하나의 모델만 학습 (순차 실행)
    if args.n_jobs == -1:
        per_model_jobs = total_cpu  # 모든 24 스레드를 단일 모델에 할당
    else:
        per_model_jobs = max(1, args.n_jobs)
    # Default to all cores when not specified
    omp_threads = args.omp_threads if args.omp_threads > 0 else total_cpu
    
    # 라이브러리 백엔드에 모든 스레드 사용 강제 설정
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(omp_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(omp_threads)
    os.environ["BLIS_NUM_THREADS"] = str(omp_threads)
    
    # OpenMP 공격적 설정 (CPU 100% 사용을 위한 핵심 설정)
    os.environ["OMP_DYNAMIC"] = "FALSE"  # 동적 스레드 조정 비활성화
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"  # Busy-wait으로 CPU 점유
    os.environ["OMP_PROC_BIND"] = "TRUE"  # 스레드를 코어에 고정
    
    # Intel MKL 최적화 (AMD Ryzen에서도 효과적)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["MKL_THREADING_LAYER"] = "GNU"  # OpenMP 사용
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    
    print(f"[INFO] CPU cores={total_cpu} (Ryzen 5900X: 12C/24T)")
    print(f"[INFO] model_parallel={parallel_workers}  omp_threads={omp_threads}  per_model_jobs={per_model_jobs}")
    print(f"[INFO] OMP_WAIT_POLICY=ACTIVE (CPU를 100% 점유합니다)")

    def run_jobs(job_funcs):
        if not job_funcs:
            return
        # 순차 실행으로 각 모델이 모든 CPU 스레드를 독점
        results = [job() for job in job_funcs]
        for logs in results:
            for line in logs:
                print(line)

    if args.mode in ("both", "binary"):
        print("\n=== Binary classification (BENIGN vs ATTACK) ===")
        # 모든 CPU를 활용하여 학습
        per_model_jobs = total_cpu

        # Logistic Regression (balanced, larger iteration budget)
        X_lr, y_lr = stratified_limit(Xtr, ytr_bin, limit_common, args.seed)
        print(f"[LR ] train size: {len(y_lr):,}")
        def job_lr():
            logs = []
            t0 = time.perf_counter()
            lr = LogisticRegression(
                max_iter=3000,
                solver="saga",
                penalty="l2",
                C=1.0,
                n_jobs=per_model_jobs,
                class_weight="balanced",
                random_state=args.seed,
                verbose=1,  # 진행상황 표시
            )
            # loky 백엔드: GIL 우회, 진정한 병렬 처리
            with parallel_backend("loky", n_jobs=per_model_jobs):
                lr.fit(X_lr, y_lr)
            lr_pred = lr.predict(Xte)
            lr_acc = accuracy_score(yte_bin, lr_pred)
            logs.append(f"[LR ] acc={lr_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
            dump(lr, MODEL_DIR / "lr_pc_bin.joblib")
            write_meta("lr_pc_bin.meta.json", {
                "type": "binary",
                "algo": "LogisticRegression(saga,balanced)",
                "train_samples": len(y_lr),
                "test_acc": float(lr_acc),
            })
            logs.append("[LR ] report\n" + classification_report(yte_bin, lr_pred, target_names=["BENIGN", "ATTACK"], digits=4))
            if args.cv and args.cv > 1:
                tcv = time.perf_counter()
                base_lr = LogisticRegression(
                    max_iter=3000,
                    solver="saga",
                    penalty="l2",
                    C=1.0,
                    n_jobs=per_model_jobs,
                    class_weight="balanced",
                    random_state=args.seed,
                )
                with parallel_backend("loky", n_jobs=per_model_jobs):
                    lr_cv = cross_val_score(
                        base_lr,
                        X_lr,
                        y_lr,
                        cv=args.cv,
                        n_jobs=per_model_jobs,
                    ).mean()
                logs.append(f"[LR ] cv{args.cv}={lr_cv:.4f}  elapsed={time.perf_counter() - tcv:.2f}s")
            return logs

        # Support Vector Machine (RBF kernel)
        X_svm, y_svm = stratified_limit(Xtr, ytr_bin, limit_svm, args.seed)
        print(f"[SVM] train size: {len(y_svm):,}")
        def job_svm():
            logs = []
            t0 = time.perf_counter()
            svm = SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=args.seed,
                cache_size=2000,  # 2GB 캐시
                verbose=True,  # 진행상황 표시
            )
            # loky 백엔드로 멀티프로세싱
            with parallel_backend("loky", n_jobs=per_model_jobs):
                svm.fit(X_svm, y_svm)
            svm_pred = svm.predict(Xte)
            svm_acc = accuracy_score(yte_bin, svm_pred)
            logs.append(f"[SVM] acc={svm_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
            dump(svm, MODEL_DIR / "svm_pc_bin.joblib")
            write_meta("svm_pc_bin.meta.json", {
                "type": "binary",
                "algo": "SVC(rbf,balanced)",
                "train_samples": len(y_svm),
                "test_acc": float(svm_acc),
            })
            logs.append("[SVM] report\n" + classification_report(yte_bin, svm_pred, target_names=["BENIGN", "ATTACK"], digits=4))
            if args.cv and args.cv > 1:
                tcv = time.perf_counter()
                base_svm = SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=args.seed,
                    cache_size=2000,
                )
                with parallel_backend("loky", n_jobs=per_model_jobs):
                    svm_cv = cross_val_score(
                        base_svm,
                        X_svm,
                        y_svm,
                        cv=args.cv,
                        n_jobs=per_model_jobs,
                    ).mean()
                logs.append(f"[SVM] cv{args.cv}={svm_cv:.4f}  elapsed={time.perf_counter() - tcv:.2f}s")
            return logs

        run_jobs([job_lr, job_svm])

    if args.mode in ("both", "multi"):
        print("\n=== Multiclass classification (Attack Type) ===")
        # 모든 CPU를 활용하여 학습
        per_model_jobs = total_cpu

        # Random Forest
        X_rf, y_rf = stratified_limit(Xtr, ytr_mc, limit_common, args.seed)
        print(f"[RF ] train size: {len(y_rf):,}")
        def job_rf():
            logs = []
            t0 = time.perf_counter()
            rf = RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=per_model_jobs,
                random_state=args.seed,
                verbose=2,  # 진행상황 상세 표시
                warm_start=False,
            )
            # loky 백엔드로 모든 스레드 활용
            with parallel_backend("loky", n_jobs=per_model_jobs):
                rf.fit(X_rf, y_rf)
            rf_pred = rf.predict(Xte)
            rf_acc = accuracy_score(yte_mc, rf_pred)
            logs.append(f"[RF ] acc={rf_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
            dump(rf, MODEL_DIR / "rf_pc_multi.joblib")
            write_meta("rf_pc_multi.meta.json", {
                "type": "multiclass",
                "algo": "RandomForestClassifier",
                "train_samples": len(y_rf),
                "test_acc": float(rf_acc),
            })
            logs.append("[RF ] report\n" + classification_report(yte_mc, rf_pred, digits=4))
            if args.cv and args.cv > 1:
                tcv = time.perf_counter()
                with parallel_backend("loky", n_jobs=per_model_jobs):
                    rf_cv = cross_val_score(rf, X_rf, y_rf, cv=args.cv, n_jobs=per_model_jobs).mean()
                logs.append(f"[RF ] cv{args.cv}={rf_cv:.4f}  elapsed={time.perf_counter() - tcv:.2f}s")
            return logs

        print("\n[+] Decision Tree (multiclass)")
        X_dt, y_dt = stratified_limit(Xtr, ytr_mc, limit_dt, args.seed)
        print(f"[DT ] train size: {len(y_dt):,}")
        def job_dt():
            logs = []
            t0 = time.perf_counter()
            dt = DecisionTreeClassifier(
                max_depth=14,
                min_samples_split=4,
                random_state=args.seed,
                splitter="best",
            )
            # DecisionTree는 병렬화 제한적이지만 NumPy 연산은 병렬화
            dt.fit(X_dt, y_dt)
            dt_pred = dt.predict(Xte)
            dt_acc = accuracy_score(yte_mc, dt_pred)
            logs.append(f"[DT ] acc={dt_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
            dump(dt, MODEL_DIR / "dt_pc_multi.joblib")
            write_meta("dt_pc_multi.meta.json", {
                "type": "multiclass",
                "algo": "DecisionTreeClassifier",
                "train_samples": len(y_dt),
                "test_acc": float(dt_acc),
            })
            logs.append("[DT ] report\n" + classification_report(yte_mc, dt_pred, digits=4))
            return logs

        print("\n[+] K Nearest Neighbors (multiclass)")
        X_knn, y_knn = stratified_limit(Xtr, ytr_mc, limit_knn, args.seed)
        print(f"[KNN] train size: {len(y_knn):,}")
        def job_knn():
            logs = []
            t0 = time.perf_counter()
            knn = KNeighborsClassifier(
                n_neighbors=10,
                algorithm="auto",
                n_jobs=per_model_jobs,
                leaf_size=30,
            )
            # loky 백엔드로 병렬 처리
            with parallel_backend("loky", n_jobs=per_model_jobs):
                knn.fit(X_knn, y_knn)
            knn_pred = knn.predict(Xte)
            knn_acc = accuracy_score(yte_mc, knn_pred)
            logs.append(f"[KNN] acc={knn_acc:.4f}  elapsed={time.perf_counter() - t0:.2f}s")
            dump(knn, MODEL_DIR / "knn_pc_multi.joblib")
            write_meta("knn_pc_multi.meta.json", {
                "type": "multiclass",
                "algo": "KNeighborsClassifier",
                "train_samples": len(y_knn),
                "test_acc": float(knn_acc),
            })
            logs.append("[KNN] report\n" + classification_report(yte_mc, knn_pred, digits=4))
            return logs

        run_jobs([job_rf, job_dt, job_knn])


if __name__ == "__main__":
    main()
