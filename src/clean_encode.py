"""
# src/clean_encode.py
from pathlib import Path
import pandas as pd, numpy as np, json

BASE = Path(__file__).resolve().parents[1]
SRC = BASE/'data'/'interim'/'cicids_merged.csv'
DST = BASE/'data'/'interim'/'cicids_clean.csv'
MAP_OUT = BASE/'data'/'interim'/'attack_map.json'
DST.parent.mkdir(parents=True, exist_ok=True)

attack_map = {
    "BENIGN":"BENIGN", "DDoS":"DDoS",
    "DoS Hulk":"DoS", "DoS slowloris":"DoS", "DoS Slowhttptest":"DoS", "DoS GoldenEye":"DoS",
    "PortScan":"Port Scan", "FTP-Patator":"Brute Force", "SSH-Patator":"Brute Force",
    "Bot":"Bot",
    "Web Attack - Brute Force":"Web Attack", "Web Attack - XSS":"Web Attack",
    "Web Attack - Sql Injection":"Web Attack",
    "Infiltration":"Infiltration", "Heartbleed":"Heartbleed"
}

def main():
    df = pd.read_csv(SRC, low_memory=False)
    # 결측 보완 (대표 컬럼 예시)
    for col in ['Flow Bytes/s','Flow Packets/s']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # 나머지 결측: 수치=중앙값, 범주=최빈값
    for c in df.columns:
        if df[c].isna().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0])

    df['Attack Type'] = df['Label'].map(attack_map)
    df = df.dropna(subset=['Attack Type'])  # 매핑 안된 라벨 제외
    df.to_csv(DST, index=False)
    MAP_OUT.write_text(json.dumps(attack_map, indent=2, ensure_ascii=False))
    print('cleaned:', df.shape, '->', DST)

if __name__ == "__main__":
    main()

"""

# src/clean_encode.py
from pathlib import Path
import pandas as pd
import numpy as np
import json
import unicodedata
import sys

BASE = Path(__file__).resolve().parents[1]

# IO paths (keep in interim to match your current pipeline)
SRC = BASE / 'data' / 'interim' / 'cicids_merged.csv'
DST = BASE / 'data' / 'interim' / 'cicids_clean.csv'
MAP_OUT = BASE / 'data' / 'interim' / 'attack_map.json'
DST.parent.mkdir(parents=True, exist_ok=True)

# Label normalization to be robust to weird hyphens / mojibake (e.g., '�', en-dash)
def _normalize_label(s: str) -> str:
    if pd.isna(s):
        return s
    s = unicodedata.normalize('NFKC', str(s)).strip()
    s = s.replace('�', '-').replace('–', '-').replace('—', '-')
    # normalize common variants
    s = s.replace('Web Attack - Sql Injection', 'Web Attack - SQL Injection')
    return s

attack_map = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS GoldenEye": "DoS",
    "PortScan": "Port Scan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Bot",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - SQL Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Helper to robustly find the label column
def _find_label_column(df: pd.DataFrame) -> str:
    # Try common exact name
    if 'Label' in df.columns:
        return 'Label'
    # Try case/whitespace-insensitive match
    for c in df.columns:
        if c.strip().lower() == 'label':
            return c
    # Try partial matches containing 'label'
    for c in df.columns:
        if 'label' in c.strip().lower():
            return c
    # As a last resort, some dumps use 'class' as target
    for alt in ('Class', 'class', 'target', 'Target', 'y'):
        if alt in df.columns:
            return alt
        for c in df.columns:
            if c.strip().lower() == alt.lower():
                return c
    # Not found -> print columns and exit with error
    print("[ERROR] Could not find label column. Available columns:", list(df.columns))
    raise KeyError("Label column not found")

def main():
    print(f"[INFO] load -> {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    print("[INFO] shape before:", df.shape)

    # 0) Trim column names
    df.columns = [c.strip() for c in df.columns]

    # 1) Normalize label text and map to Attack Type
    label_col = _find_label_column(df)
    df[label_col] = df[label_col].apply(_normalize_label)
    df['Attack Type'] = df[label_col].map(attack_map)

    # drop rows that failed to map
    before = len(df)
    df = df.dropna(subset=['Attack Type'])
    if len(df) != before:
        print(f"[INFO] dropped unmapped labels: {before} -> {len(df)} (label_col='{label_col}')")

    # 2) Drop id/time-like columns if present (they add leakage / high cardinality)
    drop_cols = [c for c in ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print("[INFO] dropped id/time cols:", drop_cols)

    # 3) Replace +/- inf with NaN in numeric columns
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # 4) Try converting object columns that are numeric-like into numbers
    #    (so that medians etc. work consistently)
    for c in df.columns:
        if c in ('Label', 'Attack Type'):
            continue
        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors='ignore')
            df[c] = converted

    # 5) Fill missing values:
    #    - representative continuous features first (if present)
    for col in ['Flow Bytes/s', 'Flow Packets/s']:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median(skipna=True)
            df[col] = df[col].fillna(med)

    #    - numeric columns: fill with median
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median(skipna=True))

    #    - non-numeric columns (except labels): fill with mode if needed
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            mode = df[c].mode(dropna=True)
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])

    # 6) Drop zero-variance columns (no information)
    nunique = df.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        df.drop(columns=zero_var, inplace=True)
        print("[INFO] dropped zero-variance cols:", zero_var)

    # 7) Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) != before:
        print(f"[INFO] drop_duplicates: {before} -> {len(df)}")

    # 8) Downcast numeric dtypes to save memory
    for c in df.columns:
        if c in ('Label', 'Attack Type'):
            continue
        col = df[c]
        if pd.api.types.is_float_dtype(col):
            if col.min(skipna=True) > np.finfo(np.float32).min and col.max(skipna=True) < np.finfo(np.float32).max:
                df[c] = col.astype(np.float32)
        elif pd.api.types.is_integer_dtype(col):
            if col.min(skipna=True) > np.iinfo(np.int32).min and col.max(skipna=True) < np.iinfo(np.int32).max:
                df[c] = col.astype(np.int32)

    # 9) Save outputs
    df.to_csv(DST, index=False)
    MAP_OUT.write_text(json.dumps(attack_map, indent=2, ensure_ascii=False))
    print(f"[DONE] cleaned: {df.shape} -> {DST}")

if __name__ == "__main__":
    main()