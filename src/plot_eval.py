# src/plot_eval.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# >>> 비-GUI 백엔드로 설정 (show()가 block 안 하도록)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc, precision_recall_curve, classification_report
)

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST = DATA_DIR / "test.csv"

def load_xy(path: Path):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("PC")]]
    y_mc = df["Attack Type"]
    y_bin = (y_mc != "BENIGN").astype(int)
    return X, y_mc, y_bin

def plot_confusion(ax, y_true, y_pred, title, labels=None, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

def bin_roc_pr(ax_roc, ax_pr, y_true, score, label):
    # score: 확률 또는 decision_function 점수
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2%})")
    ax_roc.plot([0,1],[0,1],"--",color="gray")
    ax_roc.set_xlim([-0.01,1.01]); ax_roc.set_ylim([0,1.01])
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")

    precision, recall, _ = precision_recall_curve(y_true, score)
    ax_pr.plot(recall, precision, label=label)
    ax_pr.set_xlim([0,1]); ax_pr.set_ylim([0,1.01])
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-sample", type=int, default=150_000,
                    help="ROC/PR/CM 플롯에 사용할 최대 샘플 수(무작위). 0이면 전체 사용.")
    args = ap.parse_args()

    Xte, yte_mc, yte_bin = load_xy(TEST)

    # ====== Binary models ======
    lr_path = MODEL_DIR / "lr_bin.joblib"
    svm_path = MODEL_DIR / "svm_bin.joblib"

    # 플롯용 샘플 인덱스 (옵션)
    if args.plot_sample and args.plot_sample < len(yte_bin):
        idx = np.random.default_rng(0).choice(len(yte_bin), size=args.plot_sample, replace=False)
        Xte_plot = Xte.iloc[idx]
        yte_bin_plot = yte_bin.iloc[idx]
        yte_mc_plot  = yte_mc.iloc[idx]
    else:
        Xte_plot = Xte
        yte_bin_plot = yte_bin
        yte_mc_plot  = yte_mc

    if lr_path.exists() and svm_path.exists():
        lr = load(lr_path)
        svm = load(svm_path)

        lr_pred = lr.predict(Xte_plot)
        svm_pred = svm.predict(Xte_plot)

        # 확률/점수
        lr_score  = lr.predict_proba(Xte_plot)[:,1]        # LR은 확률 빠름
        # SVM은 decision_function 사용 (predict_proba보다 훨씬 빠름)
        svm_score = svm.decision_function(Xte_plot)

        # Confusion Matrices (Binary)
        fig, axs = plt.subplots(1, 2, figsize=(12,4))
        plot_confusion(axs[0], yte_bin_plot, lr_pred, "Logistic Regression", labels=[0,1])
        plot_confusion(axs[1], yte_bin_plot, svm_pred, "Support Vector Machine", labels=[0,1])
        fig.suptitle("Binary Confusion Matrices")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_confmat.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ROC & PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
        bin_roc_pr(ax1, ax2, yte_bin_plot, lr_score,  "Logistic Regression")
        bin_roc_pr(ax1, ax2, yte_bin_plot, svm_score, "Support Vector Machine")
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_roc_pr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Reports (터미널 출력 + 파일 저장)
        rep_lr  = classification_report(yte_bin_plot, lr_pred,  target_names=["BENIGN","ATTACK"], digits=4)
        rep_svm = classification_report(yte_bin_plot, svm_pred, target_names=["BENIGN","ATTACK"], digits=4)
        print("\n[LR] report\n",  rep_lr)
        print("\n[SVM] report\n", rep_svm)
        (OUT_DIR / "binary_report_LR.txt").write_text(rep_lr)
        (OUT_DIR / "binary_report_SVM.txt").write_text(rep_svm)

    # ====== Multiclass model ======
    rf_path = MODEL_DIR / "rf_multi.joblib"
    if rf_path.exists():
        rf = load(rf_path)
        rf_pred = rf.predict(Xte_plot)

        labels = sorted(yte_mc_plot.unique())
        # Confusion Matrix (Multiclass)
        fig, ax = plt.subplots(1,1, figsize=(10,7))
        plot_confusion(ax, yte_mc_plot, rf_pred, "Random Forest (Multiclass)", labels=labels, normalize=None)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "multi_confmat.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        rep_rf = classification_report(yte_mc_plot, rf_pred, digits=4)
        print("\n[RF] report\n", rep_rf)
        (OUT_DIR / "multi_report_RF.txt").write_text(rep_rf)

if __name__ == "__main__":
    main()