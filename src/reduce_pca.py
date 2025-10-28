# src/reduce_pca.py
from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from joblib import dump
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "data" / "interim" / "cicids_clean.csv"
OUT_DIR = BASE / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DST_CSV = OUT_DIR / "cicids_pca.csv"
SCALER_OUT = OUT_DIR / "scaler.joblib"
IPCA_OUT = OUT_DIR / "ipca.joblib"
NUMERIC_COLS_OUT = OUT_DIR / "numeric_columns.json"
META_OUT = OUT_DIR / "pca_meta.json"

LABEL_COL = "Attack Type"

def iter_batches(arr: np.ndarray, batch_size: int):
    """Yield contiguous batches from a numpy array."""
    n = arr.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield arr[start:end]

def main(n_components: int, batch_size: int, limit_rows: int | None = None):
    print(f"[INFO] load -> {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    if limit_rows is not None:
        df = df.iloc[:limit_rows].copy()
        print(f"[INFO] using head rows: {limit_rows}")

    if LABEL_COL not in df.columns:
        raise KeyError(f"Missing label column '{LABEL_COL}' in {SRC}. Available: {list(df.columns)[:10]}...")

    # 1) 숫자 피처만 선택 (라벨 제외)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # 혹시 결측이 남아있다면 안전 차원에서 0으로 대체(클린 단계에서 대부분 처리됨)
    X = df[num_cols].fillna(0).to_numpy(dtype=np.float32)
    y = df[LABEL_COL].values
    print(f"[INFO] numeric features: {len(num_cols)}  total rows: {X.shape[0]}")

    # 2) StandardScaler (배치 적합)
    scaler = StandardScaler(copy=False)  # in-place scale to save memory
    # partial_fit이 없어 한 번에 fit → 매우 큰 경우엔 랜덤 샘플로 학습 후 전체 transform도 가능
    print("[INFO] fitting StandardScaler...")
    scaler.fit(X)
    print("[INFO] transforming (scaling)...")
    X = scaler.transform(X)

    # 3) Incremental PCA (partial_fit로 큰 데이터 처리)
    if n_components <= 0 or n_components > X.shape[1]:
        n_components = X.shape[1] // 2 or 1
        print(f"[WARN] adjusted n_components -> {n_components}")

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    print(f"[INFO] fitting IncrementalPCA: n_components={n_components}, batch_size={batch_size}")

    # partial_fit
    for batch in tqdm(iter_batches(X, batch_size), total=(len(X) + batch_size - 1)//batch_size, desc="partial_fit"):
        ipca.partial_fit(batch)

    # 설명 분산 합계
    evr_sum = float(np.sum(ipca.explained_variance_ratio_))
    print(f"[INFO] explained variance retained: {evr_sum:.2%}")

    # 4) transform with batches & write out
    print("[INFO] transforming to principal components and saving CSV...")
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    # 미리 파일 생성(헤더)
    with open(DST_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(pc_names + [LABEL_COL]) + "\n")

    # 배치 단위로 변환 & append
    row_written = 0
    for batch in tqdm(iter_batches(X, batch_size), total=(len(X) + batch_size - 1)//batch_size, desc="transform+save"):
        PC = ipca.transform(batch)
        # 해당 배치의 y
        y_batch = y[row_written:row_written + PC.shape[0]]
        row_written += PC.shape[0]
        out_block = pd.DataFrame(PC, columns=pc_names)
        out_block[LABEL_COL] = y_batch
        out_block.to_csv(DST_CSV, mode="a", index=False, header=False)

    # 5) 모델, 메타 저장
    dump(scaler, SCALER_OUT)
    dump(ipca, IPCA_OUT)
    NUMERIC_COLS_OUT.write_text(json.dumps(num_cols, indent=2, ensure_ascii=False))
    META_OUT.write_text(json.dumps({
        "n_components": n_components,
        "batch_size": batch_size,
        "explained_variance_ratio_sum": evr_sum,
        "source": str(SRC),
        "output_csv": str(DST_CSV),
        "rows": int(row_written),
        "cols_before": int(len(num_cols)),
        "cols_after": int(n_components),
    }, indent=2))

    print(f"[DONE] saved -> {DST_CSV}")
    print(f"[DONE] scaler -> {SCALER_OUT}")
    print(f"[DONE] ipca   -> {IPCA_OUT}")
    print(f"[DONE] feature list -> {NUMERIC_COLS_OUT}")
    print(f"[DONE] meta  -> {META_OUT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reduce CICIDS features with StandardScaler + IncrementalPCA")
    ap.add_argument("--components", "-k", type=int, default=0,
                    help="number of principal components (default: half of numeric features)")
    ap.add_argument("--batch-size", "-b", type=int, default=500,
                    help="batch size for IncrementalPCA (default: 500)")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="use only the first N rows (debug/quick run)")
    args = ap.parse_args()
    main(n_components=args.components, batch_size=args.batch_size, limit_rows=args.limit_rows)