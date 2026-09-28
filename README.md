# rag-ollama

Ollama를 LLM 백엔드로 사용하는 문서 ingestion, 비정형 텍스트 품질진단, pgvector 검색, 선택적 답변을 제공하는 작은 RAG API다.

## 현재 지원 범위

- 텍스트, Markdown, CSV, JSON, HTML, PDF, DOCX 업로드
- UTF-8/CP949 계열 텍스트 정규화
- 문서 중복 해시 확인
- 청킹 후 FlagEmbedding/BGE 임베딩
- PostgreSQL + pgvector 저장 및 cosine 검색
- `POST /query`에서 근거 chunk와 선택적 LLM 답변 반환
- 빈 문서, 인코딩 손상, 제어문자, 반복 줄, 짧은 문서를 진단하는 기본 품질 리포트

스캔 PDF는 OCR이 필요하고 HWP/HWPX는 별도 어댑터가 필요하다. 현재는 조용히 잘못 색인하지 않고 명시적으로 오류를 반환한다.

## 실행

Python 3.12와 uv가 필요하다.

```bash
cp -n .env.example .env
uv sync
docker compose up -d postgres
uv run rag-ollama-api
```

처음 `EMBEDDING_PROVIDER=flag`로 문서를 넣을 때 BGE 모델을 내려받는다. GPU가 있으면 `.env`에서 `EMBEDDING_DEVICE=cuda:0`, `EMBEDDING_USE_FP16=true`를 검토할 수 있다. 연결과 API만 먼저 확인하려면 `EMBEDDING_PROVIDER=hash`로 바꿀 수 있지만, hash 임베딩은 실제 의미 검색용이 아니다.

## API 예시

상태 확인:

```bash
curl http://localhost:8000/health
```

텍스트 등록:

```bash
curl -X POST http://localhost:8000/documents/text \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "휴가규정.md",
    "text": "연차휴가는 1년간 80퍼센트 이상 출근한 근로자에게 부여한다.",
    "metadata": {"department": "인사", "version": "2026-01"}
  }'
```

파일 등록:

```bash
curl -X POST http://localhost:8000/documents/file \
  -F 'file=@./sample.pdf' \
  -F 'metadata={"department":"품질"}'
```

검색/RAG:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"연차휴가 조건은 무엇인가?", "top_k": 5, "min_quality_score": 80, "use_llm": true}'
```

LLM 설정이 없으면 `answer`는 `null`이고, `context`와 `sources`를 받아 호출하는 애플리케이션에서 직접 답변을 생성할 수 있다. `LLM_BASE_URL`과 `LLM_MODEL`을 설정하면 OpenAI-compatible `/v1/chat/completions` 엔드포인트를 호출한다.

문서 품질 진단:

```bash
curl -X POST http://localhost:8000/quality/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"text":"진단할 문서 본문", "name":"sample.txt"}'
```

## 비정형 품질진단으로 확장하는 방향

현재 품질진단은 범용 baseline이다. 실제 진단 시스템에서는 다음 규칙을 `quality.py`에 도메인별 profile로 추가한다.

1. 완전성: 필수 섹션, 필수 메타데이터, 페이지/표 추출 여부
2. 유효성: 날짜·코드·수치·단위·문서 버전 형식
3. 일관성: 제목/본문 구조, 용어 표준, 같은 필드의 값 충돌
4. 유일성: 동일 파일 해시, 유사 문서, 반복 표·머리말
5. 가독성: OCR 누락, 인코딩 깨짐, 표/목록 순서, 너무 짧은 chunk
6. 추적성: 원본 파일명, 페이지 번호, chunk 위치, 처리 시각

품질 점수는 검색 결과에 같이 저장되므로, 이후에는 `quality_score >= 80` 필터나 품질이 낮은 문서 제외 정책을 검색에 적용할 수 있다. 운영 단계에서는 규칙 버전과 진단 결과를 별도 테이블로 분리해 재진단 이력도 보존하는 편이 좋다.
