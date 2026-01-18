# App Recommendation System

추천 시스템을 위한 Item Embedding 기반 머신러닝 파이프라인

## 개요

이 프로젝트는 세션 데이터를 기반으로 사용자의 클릭, 장바구니 담기, 주문 행동을 예측하는 추천 시스템입니다. Item Embedding과 Session Embedding을 활용하여 LightGBM 모델을 학습합니다.

## 주요 기능

- **Item Embedding 학습**: 세션 시퀀스로부터 아이템 임베딩 벡터 생성
- **Session Embedding 계산**: 최근 N개 아이템 임베딩의 평균으로 세션 표현
- **Feature Table 생성**: 이벤트 데이터와 임베딩 조인
- **Multi-label 학습**: Clicks, Carts, Orders 각각에 대한 예측 모델 학습
- **에러 로깅**: 처리 중 발생하는 오류를 `log/error.log`에 기록

## 프로젝트 구조

```
app_recommendation/
├── data/                    # 학습 데이터
│   ├── train.jsonl         # 학습 세션 데이터
│   └── test.jsonl          # 테스트 세션 데이터
├── src/                     # 소스 코드
│   ├── entity/             # 데이터 모델
│   │   └── session.py      # Session, Event 클래스
│   ├── data/               # 데이터 로딩
│   │   ├── file_processor.py      # 파일 청크 읽기
│   │   └── session_loader.py      # 세션 로딩
│   ├── features/           # Feature 생성
│   │   ├── embedding.py           # 임베딩 학습 및 계산
│   │   ├── feature_table.py       # Feature Table 생성
│   │   └── training_dataset.py    # 학습 데이터셋 생성
│   ├── training/           # 모델 학습
│   │   └── model_training.py      # LightGBM 모델 학습
│   ├── inference/          # 추천 생성
│   │   └── __init__.py            # Top-K 추천 생성
│   ├── api/                # Flask API
│   │   ├── app.py                 # Flask 애플리케이션
│   │   ├── __init__.py            # API 모듈
│   │   └── README.md              # API 문서
│   └── main.py             # 메인 실행 스크립트
├── feature/                # 생성된 feature 저장
├── models/                 # 학습된 모델 저장
├── output/                 # 추천 결과 저장
├── log/                    # 에러 로그
├── requirements.txt        # Python 패키지 목록
└── .devcontainer/          # 개발 환경 설정

```

## 환경 설정

### Dev Container 사용

```bash
# Dev Container에서 자동으로 실행됨
# - Azure CLI 설치
# - uv 패키지 매니저 설치
# - Python 의존성 설치 (numpy, pandas, lightgbm, scikit-learn)
```

### 수동 설치

```bash
# Python 패키지 설치
pip install numpy pandas lightgbm scikit-learn
```

## 실행 방법

### 전체 파이프라인 실행

```bash
cd src
uv run main.py
```

### 실행 단계

파이프라인은 다음 순서로 자동 실행됩니다:

1. **Item Embedding 추출**
   - 세션 시퀀스로부터 아이템 임베딩 학습
   - 출력: `feature/item_embeddings.pkl`

2. **Session Embedding 추출**
   - 최근 10개 아이템 임베딩의 평균 계산
   - 출력: `feature/session_embeddings.pkl`

3. **Item Feature Table 생성**
   - 이벤트 데이터와 Item Embedding 조인 (aid 기준)
   - 출력: `feature/item_feature_table.pkl`

4. **Session Feature Table 생성**
   - 이벤트 데이터와 Session Embedding 조인 (session 기준)
   - 출력: `feature/session_feature_table.pkl`

5. **Training Dataset 생성**
   - Feature Table 조인 및 라벨링
   - 라벨: 세션의 마지막 아이템 여부 (clicks, carts, orders)
   - 출력: `feature/training_dataset.pkl`

6. **LightGBM 모델 학습**
   - 각 타겟(clicks, carts, orders)별 Binary Classification 모델 학습
   - Train/Validation Split: 80/20
   - 출력:
     - `models/lgbm_model_clicks.pkl`
     - `models/lgbm_model_carts.pkl`
     - `models/lgbm_model_orders.pkl`

7. **Top-20 추천 생성**
   - 학습된 모델로 예측 수행
   - Clicks, Carts, Orders 스코어를 가중합 (0.1, 0.3, 0.6)
   - 세션별 상위 20개 아이템 선택
   - 출력: `output/recommendations.csv`

## Flask API 서버

학습된 모델을 사용하여 실시간 추천을 제공하는 REST API 서버를 제공합니다.

### API 서버 실행

```bash
cd src/api
uv run python app.py
```

서버는 `http://localhost:5000`에서 실행됩니다.

### 주요 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서버 상태 확인 |
| `/recommend` | POST | 단일 세션 추천 생성 |
| `/batch_recommend` | POST | 여러 세션 배치 추천 |
| `/items/<aid>/embedding` | GET | 아이템 임베딩 조회 |
| `/sessions/<session_id>/embedding` | GET | 세션 임베딩 조회 |
| `/stats` | GET | 통계 정보 조회 |

### 사용 예제

```python
import requests

# 추천 요청
response = requests.post('http://localhost:5000/recommend', json={
    "session_id": 12345,
    "events": [
        {"aid": 1001, "ts": 1659304800, "type": "clicks"},
        {"aid": 1002, "ts": 1659304900, "type": "carts"}
    ],
    "top_k": 20
})

result = response.json()
print(f"Recommendations: {result['recommendations']}")
```

자세한 API 문서는 [src/api/README.md](src/api/README.md)를 참고하세요.

## 데이터 형식

### 입력 데이터 (train.jsonl)

```json
{
  "session": 12345,
  "events": [
    {"aid": 1001, "ts": 1234567890, "type": "clicks"},
    {"aid": 1002, "ts": 1234567900, "type": "carts"},
    {"aid": 1003, "ts": 1234567910, "type": "orders"}
  ]
}
```

### Training Dataset 구조

| 컬럼 | 설명 |
|------|------|
| session | 세션 ID |
| aid | Article ID (상품 ID) |
| type | 이벤트 타입 (clicks, carts, orders) |
| ts | 타임스탬프 |
| label_clicks | Clicks 라벨 (마지막 클릭 아이템이면 1) |
| label_carts | Carts 라벨 (마지막 장바구니 아이템이면 1) |
| label_orders | Orders 라벨 (마지막 주문 아이템이면 1) |
| item_emb_0 ~ item_emb_127 | Item Embedding (128차원) |
| session_emb_0 ~ session_emb_127 | Session Embedding (128차원) |

## 설정

### 파라미터 조정 (src/main.py)

```python
max_lines = 50000          # 처리할 최대 라인 수 (None = 전체)
embedding_dim = 128        # 임베딩 차원
n_recent = 10             # 세션 임베딩 계산 시 사용할 최근 아이템 수
test_size = 0.2           # Validation 데이터 비율
top_k = 20                # 추천할 아이템 수
label_weights = {         # 스코어 가중치
    'clicks': 0.1,
    'carts': 0.3,
    'orders': 0.6
}
```

### LightGBM 파라미터 (src/model_training.py)

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    # ...
}
```

## 로그 및 모니터링

- **에러 로그**: `log/error.log`
- **진행 상황**: 콘솔에 실시간 출력
  - 10 청크(50,000 라인)마다 진행 상황 표시
  - Feature 매칭률 및 Coverage 통계 출력

## 출력 파일

| 파일 | 설명 | 크기 |
|------|------|------|
| `feature/item_embeddings.pkl` | 아이템 임베딩 딕셔너리 | ~MB |
| `feature/session_embeddings.pkl` | 세션 임베딩 딕셔너리 | ~MB |
| `feature/item_feature_table.pkl` | Item Feature Table | ~MB |
| `feature/session_feature_table.pkl` | Session Feature Table | ~MB |
| `feature/training_dataset.pkl` | 학습용 DataFrame | ~MB |
| `models/lgbm_model_*.pkl` | 학습된 LightGBM 모델 (3개) | ~MB |
| `output/recommendations.csv` | 세션별 Top-20 추천 결과 | ~KB |

### recommendations.csv 형식

```csv
session,labels
12345,1001 1002 1003 1004 ... (20 items)
12346,2001 2002 2003 2004 ... (20 items)
```

- `session`: 세션 ID
- `labels`: 추천 아이템 ID 목록 (공백으로 구분된 상위 20개)

## 성능

- **메모리 최적화**: 배치 처리로 대용량 파일 처리 가능
- **청크 단위 처리**: 5,000 라인씩 읽어서 메모리 효율성 확보
- **에러 핸들링**: 파싱 오류 발생 시 해당 라인 스킵 및 로깅

## 문제 해결

### 메모리 부족
- `main.py`의 `max_lines` 파라미터를 줄여서 처리량 감소

### 처리 속도 개선
- `chunk_size` 증가 (기본값: 5000)
- GPU 사용 LightGBM 설치

### 에러 확인
```bash
cat log/error.log
```

## 라이선스

MIT License