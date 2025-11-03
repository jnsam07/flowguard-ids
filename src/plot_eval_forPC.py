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
    f1_score,
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
    try:
        from train_cnn1d import TinyCNN1D
    except Exception:
        TinyCNN1D = None
    try:
        from train_mlp import MLP as BaseMLP
    except Exception:
        BaseMLP = None
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    DeepCNN1D = None
    LargeMLP = None
    TinyCNN1D = None
    BaseMLP = None


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR_BASE = BASE / "models"
MODEL_DIR_PC = MODEL_DIR_BASE / "pc"
MODEL_DIR_PC.mkdir(parents=True, exist_ok=True)
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


def maybe_standardize(raw: dict, X_np: np.ndarray) -> np.ndarray:
    if not raw.get("standardize", False):
        return X_np
    mean = raw.get("feature_mean")
    std = raw.get("feature_std")
    if mean is None or std is None:
        return X_np
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.size != X_np.shape[1] or std.size != X_np.shape[1]:
        return X_np
    std_safe = np.where(np.abs(std) < 1e-6, 1.0, std)
    return (X_np - mean) / std_safe


def _load_torch_state(path: Path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj, obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj, obj["state_dict"]
    return {"model_state": obj}, obj


def plot_training_history(meta_path: Path, output_path: Path):
    """
    meta.json에서 history 데이터를 읽어서 에포크별 학습 곡선 그래프 생성
    
    Args:
        meta_path: .meta.json 파일 경로
        output_path: 저장할 그래프 파일 경로
    """
    if not meta_path.exists():
        print(f"[WARN] {meta_path} not found, skipping history plot")
        return
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    if 'history' not in meta:
        print(f"[WARN] No 'history' found in {meta_path.name}, skipping")
        return
    
    history = meta['history']
    epochs = history.get('epoch', [])
    
    if not epochs:
        print(f"[WARN] Empty history in {meta_path.name}")
        return
    
    # 4개의 서브플롯: Loss, Accuracy, Precision/Recall/F1, CV Accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training Loss
    ax = axes[0, 0]
    train_loss = history.get('train_loss', [])
    if train_loss:
        ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss over Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 2. Validation Accuracy
    ax = axes[0, 1]
    val_acc = history.get('val_acc', [])
    if val_acc:
        ax.plot(epochs, val_acc, 'g-', linewidth=2, label='Val Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy over Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])
    
    # 3. Precision, Recall, F1
    ax = axes[1, 0]
    val_prec = history.get('val_precision', [])
    val_rec = history.get('val_recall', [])
    val_f1 = history.get('val_f1', [])
    
    if val_prec:
        ax.plot(epochs, val_prec, 'r-', linewidth=2, label='Precision')
    if val_rec:
        ax.plot(epochs, val_rec, 'b-', linewidth=2, label='Recall')
    if val_f1:
        ax.plot(epochs, val_f1, 'g-', linewidth=2, label='F1-Score')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics (Precision, Recall, F1)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.05])
    
    # 4. Test CV Accuracy (test.csv 사용)
    ax = axes[1, 1]
    test_cv = history.get('test_cv_acc', [])
    if test_cv:
        # None이 아닌 값만 필터링
        cv_epochs = [ep for ep, cv in zip(epochs, test_cv) if cv is not None]
        cv_scores = [cv for cv in test_cv if cv is not None]
        
        if cv_scores:
            ax.plot(cv_epochs, cv_scores, 'mo-', linewidth=2, markersize=8, label='Test CV Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('CV Accuracy')
            ax.set_title('Cross-Validation Accuracy (test.csv)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([0, 1.05])
            
            # CV 점수 텍스트로 표시
            for ep, cv in zip(cv_epochs, cv_scores):
                ax.text(ep, cv + 0.02, f'{cv:.3f}', ha='center', va='bottom', fontsize=8)
    
    model_name = meta_path.stem.replace('.meta', '')
    fig.suptitle(f'Training History: {model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] Training history plot → {output_path}")



def torch_predict_bin(
    model,
    X_np: np.ndarray,
    device: "torch.device",
    batch_size: int = 16384,
    add_channel: bool = False,
):
    model.eval()
    xs = torch.from_numpy(X_np.astype(np.float32, copy=False))
    if add_channel:
        xs = xs.unsqueeze(1)
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

    bin_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Load traditional baselines
    lr_variants = [
        (MODEL_DIR_BASE / "lr_bin.joblib", "Logistic Regression (Base)", "binary_report_LR.txt"),
    ]
    for lr_path, lr_label, lr_report_name in lr_variants:
        if lr_path.exists():
            lr = load(lr_path)
            y_pred = lr.predict(Xte)
            y_score = lr.predict_proba(Xte)[:, 1]
            bin_results[lr_label] = (y_pred, y_score)
            rep = classification_report(yte_bin, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
            (OUT_DIR / lr_report_name).write_text(rep)

    svm_variants = [
        (MODEL_DIR_BASE / "svm_bin.joblib", "Support Vector Machine (Base)", "binary_report_SVM.txt"),
    ]
    for svm_path, svm_label, svm_report_name in svm_variants:
        if svm_path.exists():
            svm = load(svm_path)
            if hasattr(svm, "decision_function"):
                y_score = svm.decision_function(Xte)
            else:
                y_score = svm.predict_proba(Xte)[:, 1]
            y_pred = svm.predict(Xte)
            bin_results[svm_label] = (y_pred, y_score)
            rep = classification_report(yte_bin, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
            (OUT_DIR / svm_report_name).write_text(rep)

    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        pc_dim = Xte.shape[1]

        # CNN variants
        cnn_variants = [
            (MODEL_DIR_BASE / "cnn1d_bin.pt", "CNN-1D (Base)", "binary_report_CNN1D.txt", TinyCNN1D, True),
            (MODEL_DIR_PC / "cnn1d_pc_bin.pt", "CNN-1D (PC)", "binary_report_CNN1D_pc.txt", DeepCNN1D, True),
        ]
        for cnn_path, cnn_label, cnn_report, cnn_cls, needs_channel in cnn_variants:
            if cnn_path.exists() and cnn_cls is not None:
                raw, state = _load_torch_state(cnn_path, device)
                channels = tuple(int(c) for c in raw.get("channels", raw.get("hidden_channels", [64, 128, 256])))
                fc_hidden = int(raw.get("fc_hidden", 512))
                kernel = int(raw.get("kernel_size", 3))
                dropout = float(raw.get("dropout", 0.25))
                model_kwargs = dict(
                    in_len=pc_dim,
                    num_classes=2,
                    binary=True,
                    channels=channels,
                )
                if cnn_cls is DeepCNN1D:
                    model_kwargs.update(dict(kernel_size=kernel, fc_hidden=fc_hidden, dropout=dropout))
                cnn_model = cnn_cls(**model_kwargs).to(device)
                cnn_model.load_state_dict(state, strict=True)
                preds, scores = torch_predict_bin(cnn_model, Xte_np, device, add_channel=needs_channel)
                bin_results[cnn_label] = (preds, scores)
                rep = classification_report(yte_bin, preds, target_names=["BENIGN", "ATTACK"], digits=4)
                (OUT_DIR / cnn_report).write_text(rep)

        # MLP variants
        mlp_variants = [
            (MODEL_DIR_BASE / "mlp_bin.pt", "MLP (Base)", "binary_report_MLP.txt", BaseMLP, False),
            (MODEL_DIR_PC / "mlp_pc_bin.pt", "MLP (PC)", "binary_report_MLP_pc.txt", LargeMLP, True),
        ]
        for mlp_path, mlp_label, mlp_report, mlp_cls, apply_standardize in mlp_variants:
            if mlp_path.exists() and mlp_cls is not None:
                raw, state = _load_torch_state(mlp_path, device)
                hidden = [int(h) for h in raw.get("hidden", [512, 256, 128])]
                dropout = float(raw.get("dropout", 0.3))
                mlp_kwargs = dict(in_dim=pc_dim, hidden=hidden, out_dim=1)
                if mlp_cls is LargeMLP:
                    mlp_kwargs["dropout"] = dropout
                mlp_model = mlp_cls(**mlp_kwargs).to(device)
                mlp_model.load_state_dict(state, strict=True)
                X_input = maybe_standardize(raw, Xte_np).astype(np.float32, copy=False) if apply_standardize else Xte_np
                preds, scores = torch_predict_bin(mlp_model, X_input, device)
                bin_results[mlp_label] = (preds, scores)
                rep = classification_report(yte_bin, preds, target_names=["BENIGN", "ATTACK"], digits=4)
                (OUT_DIR / mlp_report).write_text(rep)

    desired_order = [
        "MLP (PC)",
        "MLP (Base)",
        "CNN-1D (PC)",
        "CNN-1D (Base)",
        "Logistic Regression (Base)",
        "Support Vector Machine (Base)",
    ]
    bin_models = [(label, *bin_results[label]) for label in desired_order if label in bin_results]

    # Binary plots
    if bin_models:
        fig, axs = plt.subplots(1, len(bin_models), figsize=(6 * len(bin_models), 4))
        if len(bin_models) == 1:
            axs = [axs]
        for ax, (name, y_pred, _) in zip(axs, bin_models):
            plot_confusion(ax, yte_bin, y_pred, name, labels=[0, 1])
        fig.suptitle("Binary Confusion Matrices")
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
        ax.set_title("Binary Model Comparison")
        for i, v in enumerate(scores_b):
            ax.text(v + 0.01, i, f"{v:.3f}", ha="left", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_accuracy_bar_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        f1s = [(name, f1_score(yte_bin, pred, zero_division=0)) for name, pred, _ in bin_models]
        labels_f1, scores_f1 = zip(*f1s)
        fig, ax = plt.subplots(figsize=(9, 3))
        palette = sns.color_palette("Oranges", n_colors=len(labels_f1))
        ax.barh(labels_f1, scores_f1, color=palette)
        ax.set_xlim([0, 1])
        ax.set_xlabel("F1-Score")
        ax.set_title("Binary Model F1 Comparison")
        for i, v in enumerate(scores_f1):
            ax.text(v + 0.01, i, f"{v:.3f}", ha="left", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "binary_f1_bar_pc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Multiclass models
    mc_models: list[tuple[str, np.ndarray, float]] = []
    rf_path = MODEL_DIR_BASE / "rf_multi.joblib"
    if rf_path.exists():
        rf = load(rf_path)
        y_pred = rf.predict(Xte)
        acc = accuracy_score(yte_mc, y_pred)
        mc_models.append(("Random Forest (PC)", y_pred, acc))
        rep = classification_report(yte_mc, y_pred, digits=4)
        (OUT_DIR / "multiclass_report_RF_pc.txt").write_text(rep)

    dt_path = MODEL_DIR_BASE / "dt_multi.joblib"
    if dt_path.exists():
        dt = load(dt_path)
        y_pred = dt.predict(Xte)
        acc = accuracy_score(yte_mc, y_pred)
        mc_models.append(("Decision Tree (PC)", y_pred, acc))
        rep = classification_report(yte_mc, y_pred, digits=4)
        (OUT_DIR / "multiclass_report_DT_pc.txt").write_text(rep)

    knn_path = MODEL_DIR_BASE / "knn_multi.joblib"
    if knn_path.exists():
        knn = load(knn_path)
        y_pred = knn.predict(Xte)
        acc = accuracy_score(yte_mc, y_pred)
        mc_models.append(("KNN (PC)", y_pred, acc))
        rep = classification_report(yte_mc, y_pred, digits=4)
        (OUT_DIR / "multiclass_report_KNN_pc.txt").write_text(rep)

    cnn_multi_path = MODEL_DIR_PC / "cnn1d_pc_multi.pt"
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

    mlp_multi_path = MODEL_DIR_PC / "mlp_pc_multi.pt"
    if TORCH_AVAILABLE and mlp_multi_path.exists() and LargeMLP is not None:
        raw, state = _load_torch_state(mlp_multi_path, torch.device("cpu"))
        hidden = [int(h) for h in raw.get("hidden", [1024, 512, 256, 128])]
        dropout = float(raw.get("dropout", 0.3))
        classes = raw.get("classes")
        if classes:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            model = LargeMLP(in_dim=Xte.shape[1], hidden=hidden, out_dim=len(classes), dropout=dropout).to(device)
            model.load_state_dict(state, strict=True)
            Xte_mlp_multi = maybe_standardize(raw, Xte_np).astype(np.float32, copy=False)
            xs = torch.from_numpy(Xte_mlp_multi)
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

    # 학습 히스토리 그래프 생성 (CNN, MLP의 meta.json에서 읽기)
    print("\n[INFO] Plotting training histories...")
    for meta_root in (MODEL_DIR_PC, MODEL_DIR_BASE):
        for meta_file in meta_root.glob("*.meta.json"):
            model_name = meta_file.stem.replace('.meta', '')
            history_plot_path = OUT_DIR / f"history_{model_name}.png"
            plot_training_history(meta_file, history_plot_path)

    print(f"[DONE] Reports & figures → {OUT_DIR}")


if __name__ == "__main__":
    main()
