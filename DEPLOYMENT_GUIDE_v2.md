# 🛡️ FlowGuard IDS - 배포 및 사용 가이드

## 🚀 새로운 기능 (v2)

### 1. 모델 선택 기능
- **Stage 1 (이진 분류)**: MLP, CNN-1D 중 선택
- **Stage 2 (공격 유형 분류)**: Random Forest, Decision Tree, K-NN 중 선택
- **기본값**: MLP + Random Forest

### 2. SNS 공유 기능
- Twitter, Facebook, LinkedIn 공유
- 클립보드 복사 기능
- 분석 결과 요약 텍스트 자동 생성

### 3. 개선된 UI/UX
- 감성적인 그라데이션 디자인
- 실시간 모델 선택 반영
- 향상된 결과 시각화

---

## 💻 로컬 실행

```bash
# 1. 가상환경 활성화
cd "/Users/jsh/Library/Mobile Documents/com~apple~CloudDocs/ZOLZAK/flowguard-ids"
source .venv/bin/activate

# 2. 앱 실행
streamlit run src/FINAL_v2.py

# 3. 브라우저에서 접속
# http://localhost:8501
```

---

## 🌐 Streamlit Community Cloud 배포 (무료, HTTPS)

### Step 1: GitHub 준비

1. **GitHub에 코드 푸시**
```bash
git add .
git commit -m "Add FlowGuard IDS v2 with model selection"
git push origin main
```

2. **필수 파일 확인**
   - ✅ `requirements.txt` (Python 패키지 목록)
   - ✅ `.python-version` (Python 버전 지정)
   - ✅ `.streamlit/config.toml` (Streamlit 설정)
   - ✅ `src/FINAL_v2.py` (메인 앱)
   - ✅ `models/` 및 `models/pc/` 디렉토리
   - ✅ `data/processed/` 디렉토리

### Step 2: Streamlit Cloud 배포

1. **Streamlit Cloud 가입**
   - https://streamlit.io/cloud 접속
   - GitHub 계정으로 로그인

2. **New App 생성**
   - "New app" 버튼 클릭
   - Repository: `jnsam07/flowguard-ids` 선택
   - Branch: `main`
   - Main file path: `src/FINAL_v2.py`
   - App URL: 원하는 이름 입력 (예: `flowguard-ids`)

3. **고급 설정 (Advanced settings)**
   - Python version: 3.9
   - Secrets: 필요시 API 키 등 추가

4. **Deploy! 클릭**
   - 배포 시작 (약 5-10분 소요)
   - 완료 후 URL 생성됨

### Step 3: 배포된 앱 사용

배포가 완료되면 다음과 같은 URL을 받게 됩니다:
```
https://flowguard-ids-jnsam07.streamlit.app
```

이 URL을 누구나 접속할 수 있습니다! (HTTPS 보안 연결)

---

## 📱 외부 접근 설정 (로컬 네트워크)

### 로컬 네트워크에서 다른 기기로 접속

```bash
# 외부 접근 가능하게 실행
streamlit run src/FINAL_v2.py --server.address 0.0.0.0 --server.port 8501

# IP 주소 확인
ifconfig | grep "inet "
# 예: 192.168.1.100

# 같은 네트워크의 다른 기기에서 접속
# http://192.168.1.100:8501
```

### 방화벽 설정 (Mac)

```bash
# Python 허용
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp python
```

---

## 🔒 보안 고려사항

### 프로덕션 환경에서 추가 설정

1. **인증 추가**
```python
# Streamlit에 기본 인증 추가
import streamlit_authenticator as stauth

# config.yaml에 사용자 정보 저장
authenticator = stauth.Authenticate(...)
```

2. **HTTPS 설정**
   - Streamlit Cloud는 자동으로 HTTPS 제공
   - 자체 서버: Nginx + Let's Encrypt 사용

3. **Rate Limiting**
   - API 호출 제한
   - DoS 공격 방지

---

## 📊 사용 방법

### 1. 모델 선택
- 사이드바에서 Stage 1, Stage 2 모델 선택
- 기본값: MLP + Random Forest

### 2. 데이터 입력
- 테스트 데이터 샘플 사용
- 또는 CSV 파일 업로드

### 3. 분석 실행
- "🚀 분석 실행" 버튼 클릭
- 결과 확인

### 4. 결과 공유
- SNS 버튼으로 Twitter, Facebook, LinkedIn에 공유
- 또는 클립보드 복사

### 5. 결과 다운로드
- CSV 파일로 저장

---

## 🛠️ 트러블슈팅

### 포트가 이미 사용 중인 경우
```bash
# 기존 프로세스 종료
lsof -ti:8501 | xargs kill -9

# 다시 실행
streamlit run src/FINAL_v2.py
```

### 모델 파일을 찾을 수 없는 경우
```
FileNotFoundError: Model not found
```
- `models/` 및 `models/pc/` 디렉토리에 모델 파일이 있는지 확인
- 필요한 모델:
  - `models/pc/mlp_pc_bin.pt`
  - `models/pc/cnn1d_pc_bin.pt`
  - `models/rf_multi.joblib`
  - `models/dt_multi.joblib`
  - `models/knn_multi.joblib`

### GitHub 용량 제한 (100MB)
- 대용량 모델은 Git LFS 사용
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

---

## 📞 지원

문제가 발생하면:
1. GitHub Issues에 등록
2. Streamlit Community 포럼 참조
3. 로그 확인 (`~/.streamlit/logs/`)

---

## 🎉 배포 완료!

이제 전 세계 어디서나 FlowGuard IDS를 사용할 수 있습니다!

**배포된 앱 예시:**
- https://flowguard-ids.streamlit.app
- https://your-app-name.streamlit.app

**공유하기:**
- QR 코드 생성하여 발표 자료에 포함
- URL을 이메일, 슬랙, 카톡으로 공유
- SNS에 홍보
