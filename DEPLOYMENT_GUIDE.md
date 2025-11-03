# 🛡️ FlowGuard IDS - 2단계 침입 탐지 시스템

## 🚀 실행 방법

### 로컬에서 실행 (본인만 접근)
```bash
streamlit run src/FINAL.py
```

브라우저에서 `http://localhost:8501` 접속

### 외부에서 접근 가능하게 실행 (제3자도 접근 가능)
```bash
streamlit run src/FINAL.py --server.address 0.0.0.0 --server.port 8501
```

그 후 다른 사람들은 브라우저에서:
```
http://[당신의_IP주소]:8501
```

**IP 주소 확인 방법:**
- Mac/Linux: `ifconfig | grep "inet "`
- Windows: `ipconfig`

### 클라우드 배포 (인터넷에 공개)

#### Streamlit Cloud (무료, 가장 쉬움)
1. GitHub에 코드 푸시
2. https://streamlit.io/cloud 접속
3. "New app" 클릭하여 배포
4. 생성된 URL (예: `https://your-app.streamlit.app`) 공유

#### 방화벽 설정
로컬 네트워크에서 외부 접근을 허용하려면:
```bash
# Mac
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp python

# Linux (ufw)
sudo ufw allow 8501
```

## 📊 사용법

1. **데이터 소스 선택**
   - 테스트 데이터 샘플 사용
   - 또는 CSV 파일 업로드

2. **샘플 개수 조정**
   - 슬라이더로 1~200개 선택

3. **분석 실행**
   - "🚀 분석 실행" 버튼 클릭

4. **결과 확인**
   - 정상/공격 트래픽 통계
   - 각 트래픽별 상태 (✅ 정상 / 🚨 공격 ➜ 공격유형)
   - 공격 유형 분포 차트

5. **결과 다운로드**
   - CSV 파일로 저장 가능

## 🎨 주요 기능

- ✅ **이진 분류**: 딥러닝(MLP)으로 정상/공격 판별
- 🎯 **공격 유형 분류**: 머신러닝(SVM)으로 세부 공격 유형 탐지
- 📊 **시각화**: 감성적인 컬러 코딩과 통계 대시보드
- 💾 **데이터 다운로드**: CSV 형식으로 결과 저장
- 🌐 **외부 접근**: 네트워크를 통한 원격 접속 지원

## 🔒 보안 주의사항

외부에서 접근 가능하게 설정할 때:
- 신뢰할 수 있는 네트워크에서만 사용
- 민감한 데이터는 업로드하지 않기
- 필요시 인증 기능 추가 권장
