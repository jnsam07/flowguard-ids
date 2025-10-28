# src/merge_raw.py
from pathlib import Path
import pandas as pd, numpy as np, unicodedata, glob

RNG_SEED = 2025
BASE = Path(__file__).resolve().parents[1]
RAW = BASE/'data'/'raw'
OUT = BASE/'data'/'interim'/'cicids_merged.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

def normalize_label(s: str) -> str:
    s = unicodedata.normalize('NFKC', str(s)).replace('�', '-').strip()
    # 사례 표준화
    s = s.replace('Web Attack –', 'Web Attack -')
    return s

def main():
    paths = sorted(glob.glob(str(RAW/'*.csv')))
    frames = []
    # 공통 컬럼 교집합
    def cols(p): return [c.strip() for c in pd.read_csv(p, nrows=5, low_memory=False).columns]
    common = set(cols(paths[0]))
    for p in paths[1:]: common &= set(cols(p))
    common = sorted(common)

    for p in paths:
        df = pd.read_csv(p, usecols=common, low_memory=False, on_bad_lines='skip', encoding_errors='ignore')
        df.columns = [c.strip() for c in df.columns]
        if 'Label' in df.columns:
            df['Label'] = df['Label'].map(normalize_label)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    # 너무 식별적인 컬럼 제거
    data.drop(columns=[c for c in ['Flow ID','Src IP','Dst IP','Timestamp'] if c in data.columns],
              inplace=True, errors='ignore')
    # ∞→NaN
    num_cols = data.select_dtypes(include=np.number).columns
    data[num_cols] = data[num_cols].replace([np.inf, -np.inf], np.nan)
    # 중복 제거
    data.drop_duplicates(inplace=True)

    data.to_csv(OUT, index=False)
    print('merged:', data.shape, '->', OUT)

if __name__ == "__main__":
    main()