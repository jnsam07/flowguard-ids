# src/plot_eval.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

# 비-GUI 백엔드 (터미널 실행 시 block 방지)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc, precision_recall_curve, classification_report,
    accuracy_score
)

# ------- Optional: Torch (for CNN/MLP) -------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------------- Paths ----------------
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST = DATA_DIR / "test.csv"


# ---------------- Data Loaders ----------------
def load_xy(path: Path):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("PC")]]
    y_mc = df["Attack Type"]
    y_bin = (y_mc != "BENIGN").astype(int)
    return X, y_mc, y_bin


# ---------------- Plot Utils ----------------
def plot_confusion(ax, y_true, y_pred, title, labels=None, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d",
        cmap="Blues", cbar=False, ax=ax,
        xticklabels=labels, yticklabels=labels
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)


def bin_roc_pr(ax_roc, ax_pr, y_true, score, label):
    """score: probability or decision score."""
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2%})")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlim([-0.01, 1.01]); ax_roc.set_ylim([0, 1.01])
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")

    precision, recall, _ = precision_recall_curve(y_true, score)
    ax_pr.plot(recall, precision, label=label)
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.01])
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")


def report_to_matrix(rep_dict: dict, label_order: list[str]) -> np.ndarray:
    """classification_report(output_dict=True) -> heatmap용 3xN 매트릭스"""
    prec = [rep_dict[lab]["precision"] for lab in label_order]
    rec  = [rep_dict[lab]["recall"]    for lab in label_order]
    f1   = [rep_dict[lab]["f1-score"]  for lab in label_order]
    return np.array([prec, rec, f1])


# ---------------- Torch models (for saved .pt) ----------------
# --------- Helper: load various checkpoint formats ---------
def _load_torch_state(ckpt_path, device):
    """Return a state_dict regardless of how it was saved.
    Supports: torch.save(model.state_dict()),
              torch.save({'model_state': model.state_dict(), ...}),
              torch.save({'state_dict': model.state_dict(), ...}),
              torch.save(model)  # full module
    """
    obj = torch.load(ckpt_path, map_location=device)
    # If a full module was saved, extract its state_dict
    try:
        import torch.nn as nn  # local import to avoid hard torch dep at import time
        if isinstance(obj, nn.Module):
            return obj.state_dict()
    except Exception:
        pass

    if isinstance(obj, dict):
        if 'model_state' in obj and isinstance(obj['model_state'], dict):
            return obj['model_state']
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
    return obj  # assume it's already a raw state_dict
def build_cnn1d(in_ch: int, hidden: int = 64, out_bin: bool = True):
    """간단 1D CNN (train_cnn1d.py의 구조와 호환되도록 최소 정의)"""
    import torch.nn as nn
    out_dim = 1 if out_bin else 2
    model = nn.Sequential(
        nn.Conv1d(1, hidden, kernel_size=3, padding=1),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(hidden, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, out_dim)
    )
    return model


def build_mlp(in_dim: int, hidden1: int = 128, hidden2: int = 128, out_bin: bool = True):
    import torch.nn as nn
    out_dim = 1 if out_bin else 2
    model = nn.Sequential(
        nn.Linear(in_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden2, out_dim)
    )
    return model

if TORCH_AVAILABLE:
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    try:
        from train_cnn1d import TinyCNN1D
    except Exception:
        TinyCNN1D = None

    class CNN1DCompat(nn.Module):
        """
        Compatibility CNN to load checkpoints saved with named layers:
        keys like 'conv1.*', 'bn1.*', 'conv2.*', 'bn2.*', 'fc.*'
        Architecture: Conv-BN-ReLU -> Conv-BN-ReLU -> GAP -> FC(out_dim)

        NOTE: hidden1/out_channels of conv1 and hidden2/out_channels of conv2
              can be different. We parameterize both to match checkpoints.
        """
        def __init__(self, in_len: int, hidden1: int = 64, hidden2: int = 64, out_bin: bool = True):
            super().__init__()
            out_dim = 1 if out_bin else 2
            self.conv1 = nn.Conv1d(1, hidden1, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm1d(hidden1)
            self.conv2 = nn.Conv1d(hidden1, hidden2, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm1d(hidden2)
            self.relu  = nn.ReLU()
            self.gap   = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.fc    = nn.Linear(hidden2, out_dim)

        def forward(self, x):
            # x: [N, 1, F]
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.gap(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    def build_cnn1d_compat(in_ch: int, hidden1: int = 64, hidden2: int = 64, out_bin: bool = True):
        return CNN1DCompat(in_len=in_ch, hidden1=hidden1, hidden2=hidden2, out_bin=out_bin)


def torch_predict_bin(model, X_np: np.ndarray, device: "torch.device", batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_labels, scores_prob-like) for binary models."""
    model.eval()
    xs = torch.from_numpy(X_np.astype(np.float32, copy=False))
    if xs.ndim == 2:
        xs = xs.unsqueeze(1)
    ds = TensorDataset(xs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            if logits.size(1) == 1:
                prob = torch.sigmoid(logits[:, 0])
            else:
                prob = torch.softmax(logits, dim=1)[:, 1]
            scores.append(prob.cpu().numpy())
            preds.append(prob.ge(0.5).long().cpu().numpy())
    return np.concatenate(preds), np.concatenate(scores)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-sample", type=int, default=150_000,
                    help="플롯/평가에 사용할 최대 샘플 수(무작위). 0이면 전체 사용.")
    ap.add_argument("--cv-json", type=str, default="",
                    help="멀티클래스 CV 점수를 담은 JSON(optional). 키는 모델명, 값은 점수.")
    args = ap.parse_args()

    # Load test set
    Xte_all, yte_mc_all, yte_bin_all = load_xy(TEST)

    # Sampling (speed)
    if args.plot_sample and args.plot_sample < len(yte_bin_all):
        rng = np.random.default_rng(0)
        sel = rng.choice(len(yte_bin_all), size=args.plot_sample, replace=False)
        Xte = Xte_all.iloc[sel].reset_index(drop=True)
        yte_bin = yte_bin_all.iloc[sel].reset_index(drop=True)
        yte_mc  = yte_mc_all.iloc[sel].reset_index(drop=True)
    else:
        Xte, yte_bin, yte_mc = Xte_all, yte_bin_all, yte_mc_all

    Xte_np = Xte.to_numpy(dtype=np.float32, copy=True)

    # ---------- Binary (LR & SVM) ----------
    bin_models = []
    lr_path = MODEL_DIR / "lr_bin.joblib"
    svm_path = MODEL_DIR / "svm_bin.joblib"
    if lr_path.exists():
        lr = load(lr_path)
        y_pred = lr.predict(Xte)
        y_score = lr.predict_proba(Xte)[:, 1]
        bin_models.append(("Logistic Regression", y_pred, y_score))
        rep = classification_report(yte_bin, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
        (OUT_DIR / "binary_report_LR.txt").write_text(rep)
    if svm_path.exists():
        svm = load(svm_path)
        y_pred = svm.predict(Xte)
        # 빠른 점수를 위해 decision_function 사용
        if hasattr(svm, "decision_function"):
            y_score = svm.decision_function(Xte)
        else:
            y_score = svm.predict_proba(Xte)[:, 1]
        bin_models.append(("Support Vector Machine", y_pred, y_score))
        rep = classification_report(yte_bin, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
        (OUT_DIR / "binary_report_SVM.txt").write_text(rep)

    # Deep models (optional)
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        pc_dim = Xte.shape[1]

        # CNN
        cnn_pt = MODEL_DIR / "cnn1d_bin.pt"
        if cnn_pt.exists():
            raw_obj = torch.load(cnn_pt, map_location=device)
            hidden_channels: tuple[int, int] | None = None
            if isinstance(raw_obj, dict) and "hidden_channels" in raw_obj:
                vals = raw_obj["hidden_channels"]
                if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                    hidden_channels = (int(vals[0]), int(vals[1]))
            state = _load_torch_state(cnn_pt, device)

            has_named_layers = isinstance(state, dict) and any(k.startswith("conv1.") for k in state.keys())
            if hidden_channels is None and isinstance(state, dict):
                if "conv1.weight" in state and "conv2.weight" in state:
                    hidden_channels = (int(state["conv1.weight"].shape[0]), int(state["conv2.weight"].shape[0]))
                elif "0.weight" in state:
                    hidden_channels = (int(state["0.weight"].shape[0]), int(state.get("3.weight", state["0.weight"]).shape[0]) if "3.weight" in state else int(state["0.weight"].shape[0]))

            if has_named_layers and hidden_channels is not None:
                if TinyCNN1D is not None:
                    cnn = TinyCNN1D(in_len=pc_dim, num_classes=2, binary=True, channels=hidden_channels).to(device)
                else:
                    cnn = build_cnn1d_compat(pc_dim, hidden1=hidden_channels[0], hidden2=hidden_channels[1], out_bin=True).to(device)
            elif has_named_layers:
                if TinyCNN1D is not None:
                    cnn = TinyCNN1D(in_len=pc_dim, num_classes=2, binary=True).to(device)
                else:
                    cnn = build_cnn1d_compat(pc_dim, out_bin=True, hidden1=64, hidden2=64).to(device)
            else:
                h = hidden_channels[0] if hidden_channels else 64
                cnn = build_cnn1d(pc_dim, hidden=h, out_bin=True).to(device)

            cnn.load_state_dict(state, strict=True)
            preds, scores = torch_predict_bin(cnn, Xte_np, device)
            bin_models.append(("CNN-1D", preds, scores))
            rep = classification_report(yte_bin, preds, target_names=["BENIGN", "ATTACK"], digits=4)
            (OUT_DIR / "binary_report_CNN1D.txt").write_text(rep)

        # MLP
        mlp_pt = MODEL_DIR / "mlp_bin.pt"
        if mlp_pt.exists():
            state = _load_torch_state(mlp_pt, device)
            if isinstance(state, dict):
                # strip common prefixes
                if any(k.startswith("net.") for k in state.keys()):
                    state = {k.split("net.", 1)[1]: v for k, v in state.items()}
                if any(k.startswith("module.") for k in state.keys()):
                    state = {k.split("module.", 1)[1]: v for k, v in state.items()}
            # infer hidden sizes: 0.weight -> [hidden1, in], 3.weight -> [hidden2, hidden1]
            hidden1 = 128
            hidden2 = 128
            if isinstance(state, dict):
                if "0.weight" in state:
                    try:
                        hidden1 = int(state["0.weight"].shape[0])
                    except Exception:
                        pass
                if "3.weight" in state:
                    try:
                        hidden2 = int(state["3.weight"].shape[0])
                    except Exception:
                        pass
            mlp = build_mlp(pc_dim, hidden1=hidden1, hidden2=hidden2, out_bin=True).to(device)
            try:
                mlp.load_state_dict(state, strict=True)
            except RuntimeError:
                mlp.load_state_dict(state, strict=False)
            preds, scores = torch_predict_bin(mlp, Xte.values, device)
            bin_models.append(("MLP", preds, scores))
            rep = classification_report(yte_bin, preds, target_names=["BENIGN", "ATTACK"], digits=4)
            (OUT_DIR / "binary_report_MLP.txt").write_text(rep)

    # --- Binary: Confusion Matrices
    if bin_models:
        fig, axs = plt.subplots(1, len(bin_models), figsize=(6 * len(bin_models), 4))
        if len(bin_models) == 1:
            axs = [axs]
        for ax, (name, y_pred, _) in zip(axs, bin_models):
            plot_confusion(ax, yte_bin, y_pred, name, labels=[0, 1])
        fig.suptitle("Binary Confusion Matrices")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_confmats.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ROC & PR (combined)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for name, _, y_score in bin_models:
            bin_roc_pr(ax1, ax2, yte_bin, y_score, name)
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_roc_pr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Accuracy bar (binary)
        accs = [(name, accuracy_score(yte_bin, y_pred)) for name, y_pred, _ in bin_models]
        labels_b, scores_b = zip(*accs)
        fig, ax = plt.subplots(figsize=(9, 3))
        palette = sns.color_palette("Blues", n_colors=len(labels_b))
        ax.barh(labels_b, scores_b, color=palette)
        ax.set_xlim([0, 1]); ax.set_xlabel("Accuracy Score"); ax.set_title("Binary Model Comparison")
        for i, v in enumerate(scores_b):
            ax.text(v + 0.01, i, f"{v:.3f}", ha="left", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_accuracy_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Multiclass (RF / DT / KNN) ----------
    mc_avail = []
    mc_models = {
        "Random Forest": MODEL_DIR / "rf_multi.joblib",
        "Decision Trees": MODEL_DIR / "dt_multi.joblib",
        "K Nearest Neighbours": MODEL_DIR / "knn_multi.joblib",
    }
    for name, path in mc_models.items():
        if path.exists():
            clf = load(path)
            y_pred = clf.predict(Xte)
            acc = accuracy_score(yte_mc, y_pred)
            mc_avail.append((name, y_pred, acc))

            # heatmap report
            rep = classification_report(yte_mc, y_pred, output_dict=True, zero_division=0)
            labels = sorted(yte_mc.unique())
            mat = report_to_matrix(rep, labels)
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            sns.heatmap(mat, cmap="Pastel1", annot=True, fmt=".2f",
                        xticklabels=labels, yticklabels=["Precision", "Recall", "F1-score"], ax=ax)
            ax.set_title(f"Classification Report ({name})")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"multiclass_report_{name.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if mc_avail:
        # Accuracy bar
        labels_m = [n for n, _, _ in mc_avail]
        scores_m = [a for _, _, a in mc_avail]
        fig, ax = plt.subplots(figsize=(9, 3))
        palette = sns.color_palette("Blues", n_colors=len(labels_m))
        ax.barh(labels_m, scores_m, color=palette)
        ax.set_xlim([0, 1]); ax.set_xlabel("Accuracy Score"); ax.set_title("Multi-class Model Comparison")
        for i, v in enumerate(scores_m):
            ax.text(v + 0.01, i, f"{v:.4f}", ha="left", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "multiclass_accuracy_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Optional: CV scores JSON -> bar
        if args.cv_json and Path(args.cv_json).exists():
            try:
                cv_map = json.loads(Path(args.cv_json).read_text())
                labs = [k for k in labels_m if k in cv_map]
                vals = [cv_map[k] for k in labs]
                if labs:
                    fig, ax = plt.subplots(figsize=(9, 3))
                    palette = sns.color_palette("Greens", n_colors=len(labs))
                    ax.barh(labs, vals, color=palette)
                    ax.set_xlim([0, 1]); ax.set_xlabel("Cross Validation Score")
                    ax.set_title("Multi-class CV Comparison")
                    for i, v in enumerate(vals):
                        ax.text(v + 0.01, i, f"{v:.4f}", ha="left", va="center")
                    fig.tight_layout()
                    fig.savefig(OUT_DIR / "multiclass_cv_bar.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
            except Exception:
                pass

    # ---------- Summary table ----------
    # Build a quick CSV summary (accuracy numbers) to avoid re-running heavy plots.
    rows = []
    if bin_models:
        for name, y_pred, _ in bin_models:
            rows.append({"task": "binary", "model": name, "accuracy": accuracy_score(yte_bin, y_pred)})
    if mc_avail:
        for name, _, acc in mc_avail:
            rows.append({"task": "multiclass", "model": name, "accuracy": acc})
    if rows:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(OUT_DIR / "summary_accuracy.csv", index=False)

    print(f"[DONE] Reports & figures -> {OUT_DIR}")

if __name__ == "__main__":
    main()
