# Flask API for Recommendation Service

## 시작하기

### 1. 서버 실행

```bash
cd src/api
uv run python app.py
```

서버는 `http://localhost:5000`에서 실행됩니다.

## API 엔드포인트

### 1. Health Check

**Request:**
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "embeddings_loaded": true,
  "feature_columns": 256
}
```

### 2. 단일 세션 추천

**중요:** API는 입력된 events를 기반으로 세션 임베딩을 계산하고, **새로운 아이템**을 추천합니다. 입력된 events의 아이템들은 추천 결과에서 제외됩니다.

**Request:**
```bash
POST /recommend
Content-Type: application/json

{
  "session_id": 12345,  # 참고용 (로깅/분석용, 실제로는 events로 실시간 임베딩 계산)
  "events": [
    {"aid": 1001, "ts": 1659304800, "type": "clicks"},
    {"aid": 1002, "ts": 1659304900, "type": "carts"}
  ],
  "top_k": 20,
  "weights": {
    "clicks": 0.1,
    "carts": 0.3,
    "orders": 0.6
  }
}
```

**Response:**
```json
{
  "session_id": 12345,
  "recommendations": [2001, 2002, 2003, ...],  // 새로운 아이템 (1001, 1002 제외)
  "scores": {
    "clicks": [0.8, 0.7, 0.6, ...],
    "carts": [0.9, 0.8, 0.7, ...],
    "orders": [0.95, 0.85, 0.75, ...],
    "combined": [0.92, 0.82, 0.72, ...]
  }
}
```

### 3. 배치 추천

**Request:**
```bash
POST /batch_recommend
Content-Type: application/json

{
  "sessions": [
    {
      "session_id": 12345,
      "events": [
        {"aid": 1001, "ts": 1659304800, "type": "clicks"}
      ]
    },
    {
      "session_id": 12346,
      "events": [
        {"aid": 2001, "ts": 1659304800, "type": "clicks"}
      ]
    }
  ],
  "top_k": 20,
  "weights": {
    "clicks": 0.1,
    "carts": 0.3,
    "orders": 0.6
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "session_id": 12345,
      "recommendations": [1001, 1002, ...]
    },
    {
      "session_id": 12346,
      "recommendations": [2001, 2002, ...]
    }
  ],
  "errors": [],
  "total": 2,
  "success": 2,
  "failed": 0
}
```

### 4. 아이템 임베딩 조회

**Request:**
```bash
GET /items/1001/embedding
```

**Response:**
```json
{
  "aid": 1001,
  "embedding": [0.1, 0.2, 0.3, ...],
  "dimension": 128
}
```

### 5. 세션 임베딩 조회

**Request:**
```bash
GET /sessions/12345/embedding
```

**Response:**
```json
{
  "session_id": 12345,
  "embedding": [0.1, 0.2, 0.3, ...],
  "dimension": 128
}
```

### 6. 통계 정보

**Request:**
```bash
GET /stats
```

**Response:**
```json
{
  "models": ["clicks", "carts", "orders"],
  "total_items": 150000,
  "total_sessions": 50000,
  "embedding_dimension": 128,
  "feature_dimension": 256
}
```

## 사용 예제

### Python (requests)

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

### cURL

```bash
# 추천 요청
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": 12345,
    "events": [
      {"aid": 1001, "ts": 1659304800, "type": "clicks"},
      {"aid": 1002, "ts": 1659304900, "type": "carts"}
    ],
    "top_k": 20
  }'
```

## 파라미터 설명

### /recommend, /batch_recommend

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `session_id` | int | ✅ | - | 세션 ID |
| `events` | list | ✅ | - | 이벤트 목록 |
| `top_k` | int | ❌ | 20 | 추천할 아이템 수 |
| `weights` | dict | ❌ | {clicks: 0.1, carts: 0.3, orders: 0.6} | 스코어 가중치 |

### Event 객체

| 필드 | 타입 | 설명 |
|------|------|------|
| `aid` | int | 아이템 ID |
| `ts` | int | 타임스탬프 (Unix timestamp) |
| `type` | str | 이벤트 타입 (clicks, carts, orders) |

## 에러 코드

| 상태 코드 | 설명 |
|----------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 (필수 파라미터 누락 등) |
| 404 | 리소스를 찾을 수 없음 (세션/아이템 없음) |
| 500 | 서버 내부 오류 |

## 프로덕션 배포

### Gunicorn 사용

```bash
# Gunicorn 설치
uv pip install gunicorn

# 서버 실행
cd src/api
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker 배포

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY feature/ ./feature/

WORKDIR /app/src/api
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 성능 고려사항

1. **모델 로딩**: 서버 시작 시 한 번만 로딩 (글로벌 변수 사용)
2. **배치 처리**: 여러 세션을 동시에 처리할 때는 `/batch_recommend` 사용
3. **캐싱**: 자주 요청되는 추천 결과는 Redis 등으로 캐싱 권장
4. **Workers**: Gunicorn workers 수는 CPU 코어 수에 맞게 조정 (`-w 4`)
