"""
src/split_build.py
PCA가 완료된 데이터를 train/test로 나누고, 저장하는 스크립트
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "data" / "processed" / "cicids_pca.csv"

OUT_DIR = BASE / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "train.csv"
TEST_OUT = OUT_DIR / "test.csv"

def main():
    print(f"[INFO] load -> {SRC}")
    df = pd.read_csv(SRC)

    X = df[[c for c in df.columns if c.startswith("PC")]]
    y = df["Attack Type"]                 # multi-class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,         # label 비율 유지
        random_state=42
    )

    train = X_train.copy()
    train["Attack Type"] = y_train.values

    test = X_test.copy()
    test["Attack Type"] = y_test.values

    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    print(f"[DONE] train -> {TRAIN_OUT} ({train.shape})")
    print(f"[DONE] test  -> {TEST_OUT} ({test.shape})")

if __name__ == "__main__":
    main()