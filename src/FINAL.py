# FINAL.py
# Streamlit demo for two-stage IDS:
# 1) Binary detection with deep MLP (PC-optimized)
# 2) Attack-type identification with SVM on attack traffic only

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from train_mlp_forPC import LargeMLP


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
MODEL_DIR_PC = MODEL_DIR / "pc"
ATTACK_MODEL_PATH = MODEL_DIR / "svm_attack_multi.joblib"
PC_BINARY_MODEL_PATH = MODEL_DIR_PC / "mlp_pc_bin.pt"
PC_BINARY_META_PATH = MODEL_DIR_PC / "mlp_pc_bin.meta.json"

st.set_page_config(page_title="FlowGuard IDS Demo", layout="wide")


@st.cache_resource
def load_pc_feature_columns() -> list[str]:
    sample = pd.read_csv(DATA_DIR / "test.csv", nrows=1)
    return [c for c in sample.columns if c.startswith("PC")]


@st.cache_resource
def load_binary_detector() -> Dict[str, Any]:
    if not PC_BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Binary MLP checkpoint not found: {PC_BINARY_MODEL_PATH}")
    if not PC_BINARY_META_PATH.exists():
        raise FileNotFoundError(f"Binary MLP meta not found: {PC_BINARY_META_PATH}")

    meta = json.loads(PC_BINARY_META_PATH.read_text(encoding="utf-8"))
    hidden = [int(h) for h in meta["hidden"]]
    dropout = float(meta["dropout"])

    model = LargeMLP(
        in_dim=len(load_pc_feature_columns()),
        hidden=hidden,
        out_dim=1,
        dropout=dropout,
    )
    state = torch.load(PC_BINARY_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mean = np.array(meta.get("feature_mean"), dtype=np.float32) if meta.get("feature_mean") is not None else None
    std = np.array(meta.get("feature_std"), dtype=np.float32) if meta.get("feature_std") is not None else None
    threshold = float(meta.get("best_threshold", 0.5))

    return {
        "model": model,
        "device": device,
        "standardize": bool(meta.get("standardize", False)),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "meta": meta,
    }


def _sample_attack_subset(df: pd.DataFrame, limit_per_class: int = 6000, seed: int = 42) -> pd.DataFrame:
    sampled = []
    rng = np.random.default_rng(seed)
    for label, group in df.groupby("Attack Type"):
        if limit_per_class and len(group) > limit_per_class:
            idx = rng.choice(len(group), size=limit_per_class, replace=False)
            sampled.append(group.iloc[idx])
        else:
            sampled.append(group)
    return pd.concat(sampled, axis=0, ignore_index=True)


def train_attack_svm(binary_conf: Dict[str, Any]) -> Dict[str, Any]:
    st.info("학습된 공격 유형 분류 SVM을 찾을 수 없어 새로 학습합니다. (최초 1회)")
    raw = pd.read_csv(DATA_DIR / "train.csv")
    attack_df = raw[raw["Attack Type"] != "BENIGN"].copy()
    attack_df = _sample_attack_subset(attack_df, limit_per_class=6000)

    pc_cols = load_pc_feature_columns()
    X = attack_df[pc_cols].to_numpy(dtype=np.float32)
    if binary_conf["standardize"] and binary_conf["mean"] is not None:
        std_safe = np.where(np.abs(binary_conf["std"]) < 1e-6, 1.0, binary_conf["std"])
        X = (X - binary_conf["mean"]) / std_safe
    y = attack_df["Attack Type"].astype(str).to_numpy()

    encoder = LabelEncoder().fit(y)
    y_idx = encoder.transform(y)

    svm = SVC(
        kernel="rbf",
        C=3.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(X, y_idx)

    payload = {
        "model": svm,
        "label_encoder": encoder,
        "pc_cols": pc_cols,
        "standardize": binary_conf["standardize"],
        "mean": binary_conf["mean"],
        "std": binary_conf["std"],
    }
    dump(payload, ATTACK_MODEL_PATH)
    return payload


@st.cache_resource
def load_attack_classifier() -> Dict[str, Any]:
    binary_conf = load_binary_detector()
    if ATTACK_MODEL_PATH.exists():
        payload = load(ATTACK_MODEL_PATH)
        return payload
    return train_attack_svm(binary_conf)


def preprocess_features(df: pd.DataFrame, binary_conf: Dict[str, Any]) -> np.ndarray:
    pc_cols = load_pc_feature_columns()
    missing = set(pc_cols) - set(df.columns)
    if missing:
        raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")
    X = df[pc_cols].to_numpy(dtype=np.float32)
    if binary_conf["standardize"] and binary_conf["mean"] is not None:
        std_safe = np.where(np.abs(binary_conf["std"]) < 1e-6, 1.0, binary_conf["std"])
        X = (X - binary_conf["mean"]) / std_safe
    return X


def binary_stage_predict(X: np.ndarray, binary_conf: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    model: LargeMLP = binary_conf["model"]
    device: torch.device = binary_conf["device"]
    threshold: float = binary_conf["threshold"]
    tensor = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    return preds, probs


def attack_stage_predict(
    X: np.ndarray,
    attack_conf: Dict[str, Any],
    attack_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    svm: SVC = attack_conf["model"]
    encoder: LabelEncoder = attack_conf["label_encoder"]

    attack_indices = np.where(attack_mask)[0]
    if attack_indices.size == 0:
        return np.array(["BENIGN"] * len(X)), np.zeros((len(X), len(encoder.classes_)))

    attack_samples = X[attack_indices]
    probs = svm.predict_proba(attack_samples)
    preds_idx = np.argmax(probs, axis=1)
    preds = encoder.inverse_transform(preds_idx)

    out_labels = np.array(["BENIGN"] * len(X), dtype=object)
    out_probs = np.zeros((len(X), probs.shape[1]))
    out_labels[attack_indices] = preds
    out_probs[attack_indices] = probs
    return out_labels, out_probs


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    binary_conf = load_binary_detector()
    attack_conf = load_attack_classifier()
    X = preprocess_features(df, binary_conf)
    pc_cols = load_pc_feature_columns()

    preds_bin, probs_bin = binary_stage_predict(X, binary_conf)
    attack_labels, attack_probs = attack_stage_predict(X, attack_conf, preds_bin == 1)

    # 최종 상태 결정
    final_status = []
    for i, (pred, label) in enumerate(zip(preds_bin, attack_labels)):
        if pred == 0:
            final_status.append("✅ 정상")
        else:
            final_status.append(f"🚨 공격 ➜ {label}")
    
    result = pd.DataFrame({
        "트래픽 ID": [f"Traffic #{i+1}" for i in range(len(df))],
        "상태": final_status,
        "분류": np.where(preds_bin == 1, "공격", "정상"),
        "공격 확률": (probs_bin * 100).round(2),
        "공격 유형": attack_labels,
    })
    if attack_probs.size > 0:
        class_cols = [f"확률_{cls}" for cls in attack_conf["label_encoder"].classes_]
        attack_prob_df = pd.DataFrame((attack_probs * 100).round(2), columns=class_cols)
        result = pd.concat([result, attack_prob_df], axis=1)
    return result


def main():
    st.title("🛡️ FlowGuard IDS – Two-Stage Detection System")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3 style='margin: 0; color: white;'>🎯 2단계 지능형 침입 탐지 시스템</h3>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>
                <strong>Stage 1:</strong> 딥러닝(MLP) 기반 정상/공격 이진 분류<br>
                <strong>Stage 2:</strong> 머신러닝(SVM) 기반 공격 유형 세부 분류
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    binary_conf = load_binary_detector()
    attack_conf = load_attack_classifier()
    pc_cols = load_pc_feature_columns()

    with st.sidebar:
        st.markdown("### 🔧 입력 데이터 설정")
        mode = st.radio("📂 데이터 소스", ["테스트 데이터 샘플", "CSV 업로드"])
        sample_size = st.slider("📊 샘플 개수", 1, 200, 50)
        uploaded = None
        if mode == "CSV 업로드":
            uploaded = st.file_uploader("📁 CSV 파일을 업로드하세요", type=["csv"])
        
        st.markdown("---")
        st.markdown("### 🌐 외부 접근 설정")
        st.info(
            """
            **웹에서 접근하려면:**
            
            터미널에서 다음 명령어로 실행:
            ```bash
            streamlit run src/FINAL.py --server.address 0.0.0.0 --server.port 8501
            ```
            
            그 후 브라우저에서:
            `http://your-ip-address:8501`
            """
        )

    if mode == "CSV 업로드" and uploaded is not None:
        df_input = pd.read_csv(uploaded)
        st.success(f"✅ 업로드 성공: {len(df_input):,} 건의 트래픽 데이터")
    else:
        test_df = pd.read_csv(DATA_DIR / "test.csv")
        df_input = test_df.sample(sample_size, random_state=42).reset_index(drop=True)
        st.info(f"📥 테스트 데이터에서 **{sample_size}개** 샘플을 선택했습니다.")

    if st.button("🚀 분석 실행", type="primary", use_container_width=True):
        with st.spinner("🔍 트래픽 분석 중..."):
            try:
                results = run_pipeline(df_input)
                
                st.markdown("---")
                st.markdown("### 📊 분석 결과")
                
                # 통계 메트릭
                col1, col2, col3 = st.columns(3)
                total = len(results)
                attack_count = (results["분류"] == "공격").sum()
                normal_count = total - attack_count
                attack_ratio = (attack_count / total * 100) if total > 0 else 0
                
                with col1:
                    st.metric("전체 트래픽", f"{total:,} 건", delta=None)
                with col2:
                    st.metric("✅ 정상 트래픽", f"{normal_count:,} 건", 
                             delta=f"{(normal_count/total*100):.1f}%", delta_color="normal")
                with col3:
                    st.metric("🚨 공격 트래픽", f"{attack_count:,} 건", 
                             delta=f"{attack_ratio:.1f}%", delta_color="inverse")
                
                st.markdown("---")
                
                # 감성적인 결과 테이블
                st.markdown("### 🎯 상세 탐지 결과")
                
                # 스타일 적용된 데이터프레임
                def highlight_status(row):
                    if row["분류"] == "공격":
                        return ['background-color: #ffebee; color: #c62828'] * len(row)
                    else:
                        return ['background-color: #e8f5e9; color: #2e7d32'] * len(row)
                
                # 표시할 컬럼 선택
                display_cols = ["트래픽 ID", "상태", "분류", "공격 확률", "공격 유형"]
                styled_results = results[display_cols].style.apply(highlight_status, axis=1)
                
                st.dataframe(
                    styled_results,
                    use_container_width=True,
                    height=400
                )
                
                # 공격 유형 분포 (공격이 있을 경우)
                if attack_count > 0:
                    st.markdown("---")
                    st.markdown("### 🎭 공격 유형 분포")
                    attack_types = results[results["분류"] == "공격"]["공격 유형"].value_counts()
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.bar_chart(attack_types)
                    with col2:
                        st.markdown("**탐지된 공격 유형:**")
                        for attack_type, count in attack_types.items():
                            percentage = (count / attack_count * 100)
                            st.markdown(f"- **{attack_type}**: {count}건 ({percentage:.1f}%)")
                
                # 다운로드 버튼
                st.markdown("---")
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 결과를 CSV로 다운로드", 
                    data=csv_bytes,
                    file_name="flowguard_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as exc:
                st.error(f"❌ 분석 중 오류가 발생했습니다: {exc}")
                st.exception(exc)

    with st.expander("ℹ️ 모델 정보", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🧠 Binary Detection Model (MLP)**")
            st.json(binary_conf["meta"])
        with col2:
            st.markdown("**🎯 Attack Classification Model (SVM)**")
            st.json({
                "Classes": list(attack_conf["label_encoder"].classes_),
                "Total Classes": len(attack_conf["label_encoder"].classes_)
            })


if __name__ == "__main__":
    main()

