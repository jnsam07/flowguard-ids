# FINAL_v2.py
# Enhanced Streamlit demo for two-stage IDS with model selection
# 1) Binary detection with selectable deep learning models
# 2) Attack-type identification with selectable ML models

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import torch
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Try importing CNN model if available
try:
    from train_cnn1d_forPC import DeepCNN1D
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    
try:
    from train_mlp_forPC import LargeMLP
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
MODEL_DIR_PC = MODEL_DIR / "pc"

# Available models
STAGE1_MODELS = {
    "MLP (Multi-Layer Perceptron)": "mlp_pc_bin.pt",
    "CNN-1D (Convolutional Neural Network)": "cnn1d_pc_bin.pt",
}

STAGE2_MODELS = {
    "Random Forest": "rf_multi.joblib",
    "Decision Tree": "dt_multi.joblib",
    "K-Nearest Neighbors": "knn_multi.joblib",
}

st.set_page_config(
    page_title="FlowGuard IDS - AI-Powered Network Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile sidebar visibility
st.markdown("""
<style>
    /* Mobile sidebar toggle button enhancement */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.7) !important;
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pc_feature_columns() -> list[str]:
    """Load PC feature columns from test.csv or generate default list"""
    test_path = DATA_DIR / "test.csv"
    if test_path.exists():
        sample = pd.read_csv(test_path, nrows=1)
        return [c for c in sample.columns if c.startswith("PC")]
    else:
        # Default: 35 PC features (as per CNN meta file)
        return [f"PC{i}" for i in range(1, 36)]


@st.cache_resource
def load_binary_detector(model_name: str) -> Dict[str, Any]:
    """Load binary detection model (Stage 1)"""
    model_path = MODEL_DIR_PC / model_name
    meta_path = MODEL_DIR_PC / model_name.replace(".pt", ".meta.json")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # Determine model type
    if "cnn" in model_name.lower():
        if not CNN_AVAILABLE:
            raise ImportError("CNN model class not available")
        # CNN model loading
        channels = tuple(meta.get("channels", [128, 256, 512, 256]))
        kernel_size = meta.get("kernel_size", 5)
        fc_hidden = meta.get("fc_hidden", 1024)
        dropout = meta.get("dropout", 0.35)
        in_len = meta.get("pc_features", len(load_pc_feature_columns()))
        
        model = DeepCNN1D(
            in_len=in_len,
            num_classes=1,
            binary=True,
            channels=channels,
            kernel_size=kernel_size,
            fc_hidden=fc_hidden,
            dropout=dropout,
        )
    else:
        # MLP model loading
        if not MLP_AVAILABLE:
            raise ImportError("MLP model class not available")
        hidden = [int(h) for h in meta["hidden"]]
        dropout = float(meta["dropout"])
        model = LargeMLP(
            in_dim=len(load_pc_feature_columns()),
            hidden=hidden,
            out_dim=1,
            dropout=dropout,
        )
    
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state.get("model_state", state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mean = np.array(meta.get("feature_mean"), dtype=np.float32) if meta.get("feature_mean") is not None else None
    std = np.array(meta.get("feature_std"), dtype=np.float32) if meta.get("feature_std") is not None else None
    threshold = float(meta.get("best_threshold", 0.5))

    return {
        "model": model,
        "model_type": "cnn" if "cnn" in model_name.lower() else "mlp",
        "device": device,
        "standardize": bool(meta.get("standardize", False)),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "meta": meta,
    }


@st.cache_resource
def load_attack_classifier(model_name: str) -> Dict[str, Any]:
    """Load attack classification model (Stage 2)"""
    model_path = MODEL_DIR / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model = load(model_path)
    
    # For new models (RF, DT, KNN), model is directly the classifier object
    # For old SVM model, it's a dictionary
    if isinstance(model, dict):
        # Old format (SVM trained in FINAL.py)
        if "model" not in model:
            raise ValueError("Model file does not contain 'model' key")
        if "label_encoder" not in model:
            raise ValueError("Model file does not contain 'label_encoder' key")
        return model
    
    # New format (RF, DT, KNN from train_baselines.py)
    # Use model's own classes instead of creating new label encoder
    if not hasattr(model, 'classes_'):
        raise ValueError(f"Model does not have 'classes_' attribute: {type(model)}")
    
    # Create label encoder with model's actual classes
    encoder = LabelEncoder()
    encoder.classes_ = model.classes_
    
    # Get binary detector info for standardization
    binary_conf = load_binary_detector(STAGE1_MODELS[list(STAGE1_MODELS.keys())[0]])
    
    return {
        "model": model,
        "label_encoder": encoder,
        "pc_cols": binary_conf.get("pc_cols", []),
        "standardize": binary_conf.get("standardize", False),
        "mean": binary_conf.get("mean", None),
        "std": binary_conf.get("std", None),
    }


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
    model = binary_conf["model"]
    device = binary_conf["device"]
    threshold = binary_conf["threshold"]
    model_type = binary_conf["model_type"]
    
    if model_type == "cnn":
        # CNN expects (batch, channels, features)
        tensor = torch.from_numpy(X).unsqueeze(1).to(device)
    else:
        # MLP expects (batch, features)
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
    model = attack_conf["model"]
    encoder = attack_conf["label_encoder"]

    attack_indices = np.where(attack_mask)[0]
    if attack_indices.size == 0:
        return np.array(["BENIGN"] * len(X)), np.zeros((len(X), len(encoder.classes_)))

    attack_samples = X[attack_indices]
    probs = model.predict_proba(attack_samples)
    preds_idx = np.argmax(probs, axis=1)
    preds = encoder.inverse_transform(preds_idx)

    out_labels = np.array(["BENIGN"] * len(X), dtype=object)
    out_probs = np.zeros((len(X), probs.shape[1]))
    out_labels[attack_indices] = preds
    out_probs[attack_indices] = probs
    return out_labels, out_probs


def run_pipeline(df: pd.DataFrame, stage1_model: str, stage2_model: str) -> pd.DataFrame:
    binary_conf = load_binary_detector(STAGE1_MODELS[stage1_model])
    attack_conf = load_attack_classifier(STAGE2_MODELS[stage2_model])
    X = preprocess_features(df, binary_conf)

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


def create_share_text(total: int, attack_count: int, normal_count: int, attack_types: dict) -> str:
    """Create shareable text summary"""
    attack_ratio = (attack_count / total * 100) if total > 0 else 0
    
    text = f"""🛡️ FlowGuard IDS 분석 결과

📊 전체 트래픽: {total:,}건
✅ 정상: {normal_count:,}건 ({(normal_count/total*100):.1f}%)
🚨 공격: {attack_count:,}건 ({attack_ratio:.1f}%)
"""
    
    if attack_count > 0 and attack_types:
        text += "\n🎭 탐지된 공격 유형:\n"
        for attack_type, count in attack_types.items():
            percentage = (count / attack_count * 100)
            text += f"  • {attack_type}: {count}건 ({percentage:.1f}%)\n"
    
    text += "\n🔗 FlowGuard IDS - AI 기반 네트워크 보안 시스템"
    return text


def create_share_buttons(share_text: str):
    """Create social media share buttons"""
    encoded_text = quote(share_text)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        twitter_url = f"https://twitter.com/intent/tweet?text={encoded_text}"
        st.markdown(f'<a href="{twitter_url}" target="_blank"><button style="width:100%; padding:10px; background:#1DA1F2; color:white; border:none; border-radius:5px; cursor:pointer;">🐦 Twitter</button></a>', unsafe_allow_html=True)
    
    with col2:
        # Facebook share (requires URL)
        facebook_url = f"https://www.facebook.com/sharer/sharer.php?quote={encoded_text}"
        st.markdown(f'<a href="{facebook_url}" target="_blank"><button style="width:100%; padding:10px; background:#4267B2; color:white; border:none; border-radius:5px; cursor:pointer;">📘 Facebook</button></a>', unsafe_allow_html=True)
    
    with col3:
        # LinkedIn share
        linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_text}"
        st.markdown(f'<a href="{linkedin_url}" target="_blank"><button style="width:100%; padding:10px; background:#0077B5; color:white; border:none; border-radius:5px; cursor:pointer;">💼 LinkedIn</button></a>', unsafe_allow_html=True)
    
    with col4:
        # Copy to clipboard button - Using Streamlit's built-in mechanism
        if st.button("📋 복사", key="copy_button", use_container_width=True):
            st.code(share_text, language=None)
            st.success("✅ 위 텍스트를 복사하세요!")


def main():
    # Mobile user notice (outside sidebar)
    st.info("📱 **모바일 사용자 안내:** 화면 왼쪽 상단의 버튼 (>>)** 을 눌러 설정 메뉴를 열고 닫을 수 있습니다!")
    
    st.title("🛡️ FlowGuard IDS – AI-Powered Network Security")
    
    # Detailed User Guide
    with st.expander("📖 사용 가이드 (처음 사용하시나요? 여기를 클릭하세요!)", expanded=False):
        st.markdown("### 🎯 FlowGuard IDS 사용 방법")
        
        # Step 1: Model Selection
        st.markdown("#### 1️⃣ 모델 선택 (왼쪽 사이드바)")
        st.info("""
**Stage 1 모델:** 정상/공격 트래픽을 구분하는 딥러닝 모델을 선택
- **MLP:** 빠르고 가벼운 신경망 (기본 권장)
- **CNN-1D:** 더 정확하지만 느린 합성곱 신경망

**Stage 2 모델:** 공격 유형을 세부 분류하는 머신러닝 모델을 선택
- **Random Forest:** 가장 정확한 모델 (기본 권장, 9개 공격 유형)
- **Decision Tree:** 빠른 예측 (8개 공격 유형)
- **K-NN:** 간단한 모델 (7개 공격 유형)
        """)
        
        # Step 2: Data Preparation
        st.markdown("#### 2️⃣ 데이터 준비")
        st.info("""
**두 가지 방법 중 선택:**
- **테스트 데이터 샘플:** 기본 제공되는 샘플 데이터 사용 (빠른 테스트용)
- **CSV 업로드:** 직접 준비한 네트워크 트래픽 데이터 업로드
  - 필수 컬럼: PC1, PC2, ..., PC35 (PCA 처리된 특성)
  - 선택 컬럼: Attack Type (실제 공격 유형)

**샘플 개수:** 슬라이더로 분석할 데이터 양 조절 (1~200개)
        """)
        
        # Step 3: Run Analysis
        st.markdown("#### 3️⃣ 분석 실행")
        st.info("""
**"🚀 분석 실행"** 버튼을 클릭하세요!
- Stage 1: 먼저 정상/공격 트래픽 구분
- Stage 2: 공격으로 분류된 트래픽의 공격 유형 세부 분석
- 분석 시간: 샘플 개수에 따라 수 초~수십 초 소요
        """)
        
        # Step 4: View Results
        st.markdown("#### 4️⃣ 결과 확인")
        st.info("""
분석 완료 후 다음 정보를 확인할 수 있습니다:
- **📊 요약 통계:** 전체 트래픽, 정상 트래픽, 공격 트래픽 수
- **🎯 상세 결과 테이블:** 각 트래픽의 예측 결과 (색상 코딩)
  - 🟢 초록색: 정상 트래픽 (BENIGN)
  - 🔴 빨간색: 공격 트래픽
- **📈 공격 유형 분포:** 탐지된 공격의 종류별 개수 차트
        """)
        
        # Step 5: Share & Save
        st.markdown("#### 5️⃣ 결과 공유 및 저장")
        st.info("""
- **SNS 공유:** Twitter, Facebook, LinkedIn 버튼 클릭
- **📋 복사:** 복사 버튼 클릭 후 나타나는 텍스트를 복사
- **💾 CSV 다운로드:** 상세 결과를 CSV 파일로 저장
        """)
        
        # Tips
        st.success("""
**💡 팁:**
- 처음 사용하시면 **MLP + Random Forest** 조합을 추천합니다!
- 샘플 개수는 **50개**로 시작해보세요 (빠른 분석)
- 모바일에서는 가로 모드로 보시면 더 편합니다!
- 궁금한 점은 각 옵션 옆의 **❓** 아이콘을 클릭하세요
        """)
    
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3 style='margin: 0; color: white;'>🎯 2단계 지능형 침입 탐지 시스템</h3>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>
                <strong>Stage 1:</strong> 딥러닝 기반 정상/공격 이진 분류 (MLP/CNN 선택)<br>
                <strong>Stage 2:</strong> 머신러닝 기반 공격 유형 세부 분류 (RF/DT/KNN 선택)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("### 🤖 모델 선택")
        st.info("💡 처음이시라면 기본 설정(MLP + Random Forest)을 추천합니다!")
        
        st.markdown("**Stage 1: 이진 분류 모델**")
        stage1_model = st.selectbox(
            "딥러닝 모델 선택",
            list(STAGE1_MODELS.keys()),
            index=0,  # Default: MLP
            help="💡 MLP: 빠르고 가벼움(권장) / CNN-1D: 더 정확하지만 느림"
        )
        
        st.markdown("**Stage 2: 공격 유형 분류 모델**")
        stage2_model = st.selectbox(
            "머신러닝 모델 선택",
            list(STAGE2_MODELS.keys()),
            index=0,  # Default: Random Forest
            help="💡 RF: 가장 정확(권장, 9개 유형) / DT: 빠름(8개 유형) / KNN: 간단(7개 유형)"
        )
        
        st.markdown("---")
        st.markdown("### 📂 데이터 설정")
        st.info("💡 처음이시라면 '테스트 데이터 샘플'과 샘플 개수 50개를 추천합니다!")
        mode = st.radio(
            "데이터 소스", 
            ["테스트 데이터 샘플", "CSV 업로드"],
            help="테스트 데이터 샘플: 빠른 테스트 / CSV 업로드: 직접 준비한 데이터 분석"
        )
        sample_size = st.slider(
            "📊 샘플 개수", 
            1, 200, 50,
            help="분석할 데이터 개수. 많을수록 시간이 오래 걸립니다."
        )
        uploaded = None
        if mode == "CSV 업로드":
            uploaded = st.file_uploader(
                "📁 CSV 파일을 업로드하세요", 
                type=["csv"],
                help="PC1~PC35 컬럼이 필요합니다. Attack Type 컬럼은 선택사항입니다."
            )
        
        st.markdown("---")
        st.markdown("### 🌐 배포 정보")
        st.info(
            """
            **이 앱을 온라인에 배포하려면:**
            
            1. **Streamlit Community Cloud** (무료)
               - GitHub에 코드 푸시
               - streamlit.io/cloud 접속
               - "New app" 클릭하여 배포
            
            2. **로컬에서 외부 접근**
               ```bash
               streamlit run FINAL_v2.py \\
                 --server.address 0.0.0.0
               ```
            """
        )

    # Model info display
    with st.expander("ℹ️ 선택된 모델 정보", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🧠 Stage 1 Model**")
            st.info(f"Model: {stage1_model}\n\nFile: {STAGE1_MODELS[stage1_model]}")
        with col2:
            st.markdown(f"**🎯 Stage 2 Model**")
            st.info(f"Model: {stage2_model}\n\nFile: {STAGE2_MODELS[stage2_model]}")

    # Data loading
    if mode == "CSV 업로드" and uploaded is not None:
        df_input = pd.read_csv(uploaded)
        st.success(f"✅ 업로드 성공: {len(df_input):,} 건의 트래픽 데이터")
    else:
        # Try to load test.csv, if not available create dummy data
        test_path = DATA_DIR / "test.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            df_input = test_df.sample(sample_size, random_state=42).reset_index(drop=True)
            st.info(f"📥 테스트 데이터에서 **{sample_size}개** 샘플을 선택했습니다.")
        else:
            # Create dummy test data
            st.warning("⚠️ 테스트 데이터 파일이 없습니다. 데모용 데이터를 생성합니다.")
            pc_cols = load_pc_feature_columns()
            n_samples = min(sample_size, 100)
            
            # Generate random PC features
            np.random.seed(42)
            dummy_data = {col: np.random.randn(n_samples) for col in pc_cols}
            
            # Add attack types
            attack_types = ['BENIGN', 'Bot', 'DDoS', 'DoS', 'Port Scan', 'Brute Force']
            dummy_data['Attack Type'] = np.random.choice(attack_types, n_samples)
            
            df_input = pd.DataFrame(dummy_data)
            st.info(f"🧪 **{n_samples}개** 데모 샘플을 생성했습니다. CSV를 업로드하여 실제 데이터를 분석하세요.")

    st.markdown("---")
    st.markdown("### 🚀 분석 시작")
    st.info("💡 모든 설정을 완료하셨나요? 아래 버튼을 클릭하여 분석을 시작하세요!")
    
    if st.button("🚀 분석 실행", type="primary", use_container_width=True):
        with st.spinner(f"🔍 {stage1_model} & {stage2_model} 모델로 분석 중... (잠시만 기다려주세요)"):
            try:
                results = run_pipeline(df_input, stage1_model, stage2_model)
                
                st.markdown("---")
                st.markdown("### 📊 분석 결과")
                st.success("✅ 분석이 완료되었습니다! 아래에서 결과를 확인하세요.")
                
                # Statistics
                total = len(results)
                attack_count = (results["분류"] == "공격").sum()
                normal_count = total - attack_count
                attack_ratio = (attack_count / total * 100) if total > 0 else 0
                
                st.markdown("#### 📈 요약 통계")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("전체 트래픽", f"{total:,} 건")
                with col2:
                    st.metric("✅ 정상 트래픽", f"{normal_count:,} 건", 
                             delta=f"{(normal_count/total*100):.1f}%", delta_color="normal")
                with col3:
                    st.metric("🚨 공격 트래픽", f"{attack_count:,} 건", 
                             delta=f"{attack_ratio:.1f}%", delta_color="inverse")
                
                st.markdown("---")
                
                # Results table
                st.markdown("### 🎯 상세 탐지 결과")
                st.info("💡 표 색상: 🟢 초록색 = 정상 트래픽 / 🔴 빨간색 = 공격 트래픽")
                
                def highlight_status(row):
                    if row["분류"] == "공격":
                        return ['background-color: #ffebee; color: #c62828'] * len(row)
                    else:
                        return ['background-color: #e8f5e9; color: #2e7d32'] * len(row)
                
                display_cols = ["트래픽 ID", "상태", "분류", "공격 확률", "공격 유형"]
                styled_results = results[display_cols].style.apply(highlight_status, axis=1)
                
                st.dataframe(styled_results, use_container_width=True, height=400)
                
                # Attack distribution
                attack_types_dict = {}
                if attack_count > 0:
                    st.markdown("---")
                    st.markdown("### 🎭 공격 유형 분포")
                    st.info("💡 탐지된 공격의 종류별 개수를 보여줍니다.")
                    attack_types = results[results["분류"] == "공격"]["공격 유형"].value_counts()
                    attack_types_dict = attack_types.to_dict()
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.bar_chart(attack_types)
                    with col2:
                        st.markdown("**탐지된 공격 유형:**")
                        for attack_type, count in attack_types.items():
                            percentage = (count / attack_count * 100)
                            st.markdown(f"- **{attack_type}**: {count}건 ({percentage:.1f}%)")
                
                # Share functionality
                st.markdown("---")
                st.markdown("### 📤 결과 공유")
                st.info("💡 SNS에 공유하거나, '📋 복사' 버튼을 눌러 텍스트를 확인 후 복사하세요!")
                
                share_text = create_share_text(total, attack_count, normal_count, attack_types_dict)
                
                create_share_buttons(share_text)
                
                # Download button
                st.markdown("---")
                st.markdown("### 💾 결과 저장")
                st.info("💡 분석 결과를 CSV 파일로 저장하여 나중에 다시 확인할 수 있습니다!")
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 결과를 CSV로 다운로드", 
                    data=csv_bytes, 
                    file_name=f"flowguard_results_{stage1_model.split()[0]}_{stage2_model.split()[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as exc:
                st.error(f"❌ 분석 중 오류가 발생했습니다: {exc}")
                st.exception(exc)
    else:
        # Display guide when button not clicked
        st.info("👆 위의 '📖 사용 가이드'를 펼쳐보시고, 왼쪽 사이드바에서 모델과 데이터를 선택한 후 '🚀 분석 실행' 버튼을 클릭하세요!")


if __name__ == "__main__":
    main()
