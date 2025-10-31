# src/plot_eval_forPC.py
from __future__ import annotations
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Optional Torch imports
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from train_cnn1d_forPC import DeepCNN1D
    from train_mlp_forPC import LargeMLP
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    DeepCNN1D = None
    LargeMLP = None


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models" / "pc"
OUT_DIR = BASE / "reports" / "pc"
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
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)


def bin_roc_pr(ax_roc, ax_pr, y_true, score, label):
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2%})")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([0, 1.01])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")

    precision, recall, _ = precision_recall_curve(y_true, score)
    ax_pr.plot(recall, precision, label=label)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.01])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")


def report_to_matrix(rep_dict: dict, label_order: list[str]) -> np.ndarray:
    prec = [rep_dict[label]["precision"] for label in label_order]
    rec = [rep_dict[label]["recall"] for label in label_order]
    f1 = [rep_dict[label]["f1-score"] for label in label_order]
    return np.array([prec, rec, f1])


def _load_torch_state(path: Path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj, obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj, obj["state_dict"]
    return {"model_state": obj}, obj


def torch_predict_bin(model, X_np: np.ndarray, device: "torch.device", batch_size: int = 16384):
    model.eval()
    xs = torch.from_numpy(X_np.astype(np.float32, copy=False)).unsqueeze(1)
    ds = TensorDataset(xs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, scores = [], []
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


def main():
    ap = argparse.ArgumentParser(description="Evaluate desktop-scale models and plot reports.")
    ap.add_argument("--plot-sample", type=int, default=0, help="평가/플롯에 사용할 샘플 수 (0이면 전체)")
    ap.add_argument("--cv-json", type=str, default="", help="멀티클래스 CV 점수 JSON (선택)")
    args = ap.parse_args()

    Xte_all, yte_mc_all, yte_bin_all = load_xy(TEST)
    if args.plot_sample and args.plot_sample > 0 and args.plot_sample < len(yte_bin_all):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(yte_bin_all), size=args.plot_sample, replace=False)
        Xte = Xte_all.iloc[idx].reset_index(drop=True)
        yte_mc = yte_mc_all.iloc[idx].reset_index(drop=True)
        yte_bin = yte_bin_all.iloc[idx].reset_index(drop=True)
    else:
        Xte, yte_mc, yte_bin = Xte_all, yte_mc_all, yte_bin_all

    Xte_np = Xte.to_numpy(dtype=np.float32, copy=True)

    bin_models: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Load traditional baselines
    lr_path = MODEL_DIR / "lr_pc_bin.joblib"
    if lr_path.exists():
        lr = load(lr_path)
        y_pred = lr.predict(Xte)
        y_score = lr.predict_proba(Xte)[:, 1]
        bin_models.append(("Logistic Regression (PC)", y_pred, y_score))
        rep = classification_report(yte_bin, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
        (OUT_DIR / "binary_report_LR_pc.txt").write_text(rep)

    svm_path = MODEL_DIR / "svm_pc_bin.joblib"
    if svm_path.exists():
        svm = load(svm_path)
        if hasattr(svm, "decision_function"):
            y_score = svm.decision_function(Xte)
        else:
            y_score = svm.predict_proba(Xte)[:, 1]
        y_pred = svm.predict(Xte)
        bin_models.append(("SVM (PC)", y_pred, y_score))
        rep = classification_report(yte_bin, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
        (OUT_DIR / "binary_report_SVM_pc.txt").write_text(rep)

    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        pc_dim = Xte.shape[1]

        cnn_path = MODEL_DIR / "cnn1d_pc_bin.pt"
        if cnn_path.exists() and DeepCNN1D is not None:
            raw, state = _load_torch_state(cnn_path, device)
            channels = tuple(int(c) for c in raw.get("channels", raw.get("hidden_channels", [64, 128, 256])))
            fc_hidden = int(raw.get("fc_hidden", 512))
            kernel = int(raw.get("kernel_size", 3))
            dropout = float(raw.get("dropout", 0.25))
            cnn = DeepCNN1D(
                in_len=pc_dim,
                num_classes=2,
                binary=True,
                channels=channels,
                kernel_size=kernel,
                fc_hidden=fc_hidden,
                dropout=dropout,
            ).to(device)
            cnn.load_state_dict(state, strict=True)
            preds, scores = torch_predict_bin(cnn, Xte_np, device)
            bin_models.append(("CNN-1D (PC)", preds, scores))
            rep = classification_report(yte_bin, preds, target_names=["BENIGN", "ATTACK"], digits=4)
            (OUT_DIR / "binary_report_CNN1D_pc.txt").write_text(rep)

        mlp_path = MODEL_DIR / "mlp_pc_bin.pt"
        if mlp_path.exists() and LargeMLP is not None:
            raw, state = _load_torch_state(mlp_path, device)
            hidden = [int(h) for h in raw.get("hidden", [1024, 512, 256, 128])]
            dropout = float(raw.get("dropout", 0.3))
            mlp = LargeMLP(in_dim=pc_dim, hidden=hidden, out_dim=1, dropout=dropout).to(device)
            mlp.load_state_dict(state, strict=True)
            preds, scores = torch_predict_bin(mlp, Xte_np, device)
            bin_models.append(("MLP (PC)", preds, scores))
            rep = classification_report(yte_bin, preds, target_names=["BENIGN", "ATTACK"], digits=4)
            (OUT_DIR / "binary_report_MLP_pc.txt").write_text(rep)

    # Binary plots
    if bin_models:
        fig, axs = plt.subplots(1, len(bin_models), figsize=(6 * len(bin_models), 4))
        if len(bin_models) == 1:
            axs = [axs]
        for ax, (name, y_pred, _) in zip(axs, bin_models):
            plot_confusion(ax, yte_bin, y_pred, name, labels=[0, 1])
        fig.suptitle("Binary Confusion Matrices (PC models)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_confmats_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for name, _, score in bin_models:
            bin_roc_pr(ax1, ax2, yte_bin, score, name)
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_roc_pr_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        accs = [(name, accuracy_score(yte_bin, pred)) for name, pred, _ in bin_models]
        labels_b, scores_b = zip(*accs)
        fig, ax = plt.subplots(figsize=(9, 3))
        palette = sns.color_palette("Blues", n_colors=len(labels_b))
        ax.barh(labels_b, scores_b, color=palette)
        ax.set_xlim([0, 1])
        ax.set_xlabel("Accuracy Score")
        ax.set_title("Binary Model Comparison (PC)")
        for i, v in enumerate(scores_b):
            ax.text(v + 0.01, i, f"{v:.3f}", ha="left", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_accuracy_bar_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Multiclass models
    mc_models: list[tuple[str, np.ndarray, float]] = []
    rf_path = MODEL_DIR / "rf_pc_multi.joblib"
    if rf_path.exists():
        rf = load(rf_path)
        y_pred = rf.predict(Xte)
        acc = accuracy_score(yte_mc, y_pred)
        mc_models.append(("Random Forest (PC)", y_pred, acc))
        rep = classification_report(yte_mc, y_pred, digits=4)
        (OUT_DIR / "multiclass_report_RF_pc.txt").write_text(rep)

    dt_path = MODEL_DIR / "dt_pc_multi.joblib"
    if dt_path.exists():
        dt = load(dt_path)
        y_pred = dt.predict(Xte)
        acc = accuracy_score(yte_mc, y_pred)
        mc_models.append(("Decision Tree (PC)", y_pred, acc))
        rep = classification_report(yte_mc, y_pred, digits=4)
        (OUT_DIR / "multiclass_report_DT_pc.txt").write_text(rep)

    knn_path = MODEL_DIR / "knn_pc_multi.joblib"
    if knn_path.exists():
        knn = load(knn_path)
        y_pred = knn.predict(Xte)
        acc = accuracy_score(yte_mc, y_pred)
        mc_models.append(("KNN (PC)", y_pred, acc))
        rep = classification_report(yte_mc, y_pred, digits=4)
        (OUT_DIR / "multiclass_report_KNN_pc.txt").write_text(rep)

    cnn_multi_path = MODEL_DIR / "cnn1d_pc_multi.pt"
    if TORCH_AVAILABLE and cnn_multi_path.exists() and DeepCNN1D is not None:
        raw, state = _load_torch_state(cnn_multi_path, torch.device("cpu"))
        channels = tuple(int(c) for c in raw.get("channels", [64, 128, 256]))
        fc_hidden = int(raw.get("fc_hidden", 512))
        kernel = int(raw.get("kernel_size", 3))
        dropout = float(raw.get("dropout", 0.25))
        classes = raw.get("classes")
        if classes:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            model = DeepCNN1D(
                in_len=Xte.shape[1],
                num_classes=len(classes),
                binary=False,
                channels=channels,
                kernel_size=kernel,
                fc_hidden=fc_hidden,
                dropout=dropout,
            ).to(device)
            model.load_state_dict(state, strict=True)
            xs = torch.from_numpy(Xte_np).unsqueeze(1)
            loader = DataLoader(TensorDataset(xs), batch_size=8192, shuffle=False)
            preds_list = []
            with torch.no_grad():
                for (xb,) in loader:
                    logits = model(xb.to(device))
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    preds_list.append(pred)
            preds = np.concatenate(preds_list)
            acc = accuracy_score(yte_mc, [classes[p] for p in preds])
            mc_models.append(("CNN-1D (PC)", np.array([classes[p] for p in preds]), acc))
            rep = classification_report(yte_mc, [classes[p] for p in preds], target_names=classes, digits=4)
            (OUT_DIR / "multiclass_report_CNN1D_pc.txt").write_text(rep)

    mlp_multi_path = MODEL_DIR / "mlp_pc_multi.pt"
    if TORCH_AVAILABLE and mlp_multi_path.exists() and LargeMLP is not None:
        raw, state = _load_torch_state(mlp_multi_path, torch.device("cpu"))
        hidden = [int(h) for h in raw.get("hidden", [1024, 512, 256, 128])]
        dropout = float(raw.get("dropout", 0.3))
        classes = raw.get("classes")
        if classes:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            model = LargeMLP(in_dim=Xte.shape[1], hidden=hidden, out_dim=len(classes), dropout=dropout).to(device)
            model.load_state_dict(state, strict=True)
            xs = torch.from_numpy(Xte_np)
            loader = DataLoader(TensorDataset(xs), batch_size=8192, shuffle=False)
            preds_list = []
            with torch.no_grad():
                for (xb,) in loader:
                    logits = model(xb.to(device))
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    preds_list.append(pred)
            preds = np.concatenate(preds_list)
            acc = accuracy_score(yte_mc, [classes[p] for p in preds])
            mc_models.append(("MLP (PC)", np.array([classes[p] for p in preds]), acc))
            rep = classification_report(yte_mc, [classes[p] for p in preds], target_names=classes, digits=4)
            (OUT_DIR / "multiclass_report_MLP_pc.txt").write_text(rep)

    if mc_models:
        labels = sorted(yte_mc.unique())
        fig, axs = plt.subplots(1, len(mc_models), figsize=(6 * len(mc_models), 4))
        if len(mc_models) == 1:
            axs = [axs]
        for ax, (name, y_pred, _) in zip(axs, mc_models):
            plot_confusion(ax, yte_mc, y_pred, name, labels=labels)
        fig.suptitle("Multiclass Confusion Matrices (PC models)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "multiclass_confmats_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        accs = [acc for _, _, acc in mc_models]
        names = [name for name, _, _ in mc_models]
        fig, ax = plt.subplots(figsize=(9, 3))
        palette = sns.color_palette("Purples", n_colors=len(names))
        ax.barh(names, accs, color=palette)
        ax.set_xlim([0, 1])
        ax.set_xlabel("Accuracy Score")
        ax.set_title("Multiclass Model Comparison (PC)")
        for i, v in enumerate(accs):
            ax.text(v + 0.01, i, f"{v:.4f}", ha="left", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "multiclass_accuracy_bar_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    rows = []
    for name, y_pred, _ in bin_models:
        rows.append({"task": "binary", "model": name, "accuracy": accuracy_score(yte_bin, y_pred)})
    for name, _, acc in mc_models:
        rows.append({"task": "multiclass", "model": name, "accuracy": acc})
    if rows:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(OUT_DIR / "summary_accuracy_pc.csv", index=False)

    print(f"[DONE] Reports & figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
