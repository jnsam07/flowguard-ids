# 🛡️ FlowGuard IDS - AI-Powered Network Security System

![FlowGuard IDS](https://img.shields.io/badge/AI-Network%20Security-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

2단계 지능형 침입 탐지 시스템 (Two-Stage Intrusion Detection System)

### 🤖모델 개발 과정🤖

       [Raw Data]
            │
            ▼
   1) 데이터 병합 (merge_raw.py)
         ↓
   2) 전처리 & 라벨 정제 (clean_encode.py)
         ↓
   3) 차원 축소 PCA (reduce_pca.py)
         ↓
   4) Train / Test Split (split_build.py)
         ↓
┌───────────── Model Training ────────────----─┐
│  ML (Multi-class)  |  DL (Binary Detection)  │
│  ───────────────── | ─────────────────────── │
│  RF / DT / KNN     | CNN-1D / MLP            │
└──────────────────────────────────────────────┘
            ▼
       모델 평가 (plot_eval.py)
            ▼
    최종 모델 저장 및 UI 연동 (FINAL.py)


## ✨ 주요 기능

### 🎯 2단계 탐지 시스템
- **Stage 1**: 딥러닝(DL) 기반 정상/공격 이진 분류
  - MLP (Multi-Layer Perceptron)
  - CNN-1D (Convolutional Neural Network)
  
- **Stage 2**: 머신러닝(ML) 기반 공격 유형 세부 분류
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors

### 🚀 v2 새로운 기능
- ✅ **모델 선택**: Stage 1, Stage 2 모델을 실시간으로 선택 가능
- 📤 **SNS 공유**: Twitter, Facebook, LinkedIn, 클립보드 복사
- 🎨 **감성적인 UI**: 그라데이션 디자인 및 컬러 코딩
- 📊 **향상된 시각화**: 공격 유형 분포 차트
- 💾 **CSV 다운로드**: 분석 결과 저장
- 🌐 **온라인 배포**: Streamlit Cloud 지원 (HTTPS)

## 🚀 빠른 시작

### 로컬 실행

```bash
# 1. 리포지토리 클론
git clone https://github.com/jnsam07/flowguard-ids.git
cd flowguard-ids

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 앱 실행
streamlit run src/FINAL_v2.py
```

브라우저에서 자동으로 `http://localhost:8501` 열림

### 온라인 배포 (Streamlit Cloud)

1. GitHub에 코드 푸시
2. https://streamlit.io/cloud 접속
3. "New app" 클릭
4. 리포지토리 선택 및 배포

**배포된 앱 URL**: `https://your-app-name.streamlit.app`

자세한 내용은 [DEPLOYMENT_GUIDE_v2.md](DEPLOYMENT_GUIDE_v2.md) 참조

## 📦 프로젝트 구조

```
flowguard-ids/
├── src/
│   ├── FINAL_v2.py          # 메인 Streamlit 앱 (v2)
│   ├── train_mlp_forPC.py   # MLP 모델 정의
│   ├── train_cnn1d.py       # CNN 모델 정의
│   └── ...
├── models/                   # 학습된 모델
│   ├── pc/                  # Stage 1 모델
│   │   ├── mlp_pc_bin.pt
│   │   ├── cnn1d_pc_bin.pt
│   │   └── *.meta.json
│   ├── rf_multi.joblib      # Random Forest
│   ├── dt_multi.joblib      # Decision Tree
│   └── knn_multi.joblib     # K-NN
├── data/
│   └── processed/           # 전처리된 데이터
│       ├── train.csv
│       └── test.csv
├── .streamlit/
│   └── config.toml          # Streamlit 설정
├── requirements.txt         # Python 패키지
├── .gitattributes          # Git LFS 설정
└── README.md

```

## 🎮 사용 방법

### 1️⃣ 모델 선택
사이드바에서 원하는 모델 조합 선택:
- **Stage 1**: MLP 또는 CNN-1D
- **Stage 2**: Random Forest, Decision Tree, K-NN

### 2️⃣ 데이터 입력
- **테스트 데이터**: 샘플 크기 조정 (1-200)
- **CSV 업로드**: 직접 데이터 업로드

### 3️⃣ 분석 실행
"🚀 분석 실행" 버튼 클릭

### 4️⃣ 결과 확인
- 📊 통계 메트릭 (전체/정상/공격 트래픽)
- 🎯 상세 탐지 결과 테이블 (컬러 코딩)
- 🎭 공격 유형 분포 차트

### 5️⃣ 결과 공유
- 🐦 Twitter
- 📘 Facebook
- 💼 LinkedIn
- 📋 클립보드 복사

### 6️⃣ 다운로드
💾 CSV 파일로 저장

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Deep Learning**: PyTorch (MLP, CNN-1D)
- **Machine Learning**: scikit-learn (RF, DT, K-NN)
- **Data Processing**: pandas, numpy
- **Deployment**: Streamlit Community Cloud

## 📊 데이터셋

CICIDS-2017 데이터셋 사용
- **정상 트래픽**: BENIGN
- **공격 유형**: DDoS, DoS, Port Scan, Brute Force, Bot, Web Attack, Infiltration, Heartbleed

## 🔒 보안 고려사항

- ✅ HTTPS 자동 지원 (Streamlit Cloud)
- ✅ CORS 보호
- ✅ XSRF 보호
- ⚠️ 프로덕션 환경에서는 추가 인증 권장

## 🤝 기여

기여를 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

MIT License - 자유롭게 사용 가능

## 👨‍💻 개발자

- **Author**: jnsam07
- **GitHub**: https://github.com/jnsam07
- **Project**: https://github.com/jnsam07/flowguard-ids
- **E-mail**: jnsam07@soongsil.ac.kr

## 📞 지원

- 📧 Issues: [GitHub Issues](https://github.com/jnsam07/flowguard-ids/issues)
- 📖 Documentation: [DEPLOYMENT_GUIDE_v2.md](DEPLOYMENT_GUIDE_v2.md)
- 💬 Community: Streamlit Community Forum

## 🎉 데모

**온라인 데모**: [https://flowguard-ids.streamlit.app](https://flowguard-ids.streamlit.app)

---

Made with ❤️ using Streamlit and PyTorch
