# src/merge_raw.py
from pathlib import Path
import pandas as pd
import glob

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'data' / 'raw'
OUT = BASE / 'data' / 'interim' / 'cicids_merged.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

paths = sorted(glob.glob(str(RAW / '*.csv')))
print(f"[INFO] raw csv files found = {len(paths)}")
for p in paths:
    print(" -", p)

if not paths:
    raise SystemExit("[ERROR] data/raw/*.csv 파일을 찾지 못했습니다.")

dfs = []
for p in paths:
    # 헤더 차이/깨진 행이 있어도 읽기 계속하도록 옵션 부여
    df = pd.read_csv(p, low_memory=False, on_bad_lines='skip', encoding_errors='ignore')
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print("[INFO] merged shape =", merged.shape)

merged.to_csv(OUT, index=False)
print("[DONE] saved ->", OUT)