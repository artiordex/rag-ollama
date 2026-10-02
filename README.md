# rag-ollama

개인 학습용 로컬 Hugging Face Transformers 및 RAG 실험 API다. 기본 생성 백엔드는 Transformers가 모델을 직접 불러오는 방식이다. Ollama나 vLLM 같은 OpenAI 호환 서버도 비교용 백엔드로 선택할 수 있다. 문서 진단·청킹·BGE 임베딩·pgvector 검색은 생성 모델과 분리되어 동작한다.

```text
문서 → 텍스트 추출·정규화 → 품질 진단 → chunk → BGE 임베딩 → pgvector 검색 → (선택) Transformers 답변
```

기본 실행은 API와 PostgreSQL을 `localhost`에만 노출한다. 첫 문서 등록 때 BGE 임베딩 가중치를, 첫 생성 요청 때 Hugging Face 모델 가중치를 내려받아 로컬 캐시에 저장한다. 이 프로젝트 폴더와 CLI 이름은 기존 경로와의 호환을 위해 `rag-ollama`로 유지한다.

## 시작하기

필요한 도구는 Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Compose다. 기본 Transformers 백엔드에는 Ollama 설치가 필요하지 않다. Hugging Face Hub에서 모델을 받을 수 있어야 하며, gated 모델은 `HF_TOKEN`을 설정한다.

프로젝트 폴더에서 설정 파일을 만들고 데이터베이스를 시작한다.

```bash
cp -n .env.example .env
uv sync
docker compose up -d postgres
```

기본 설정은 `Qwen/Qwen3-4B-Instruct-2507`을 Transformers로 직접 불러온다. 첫 생성 요청에서 모델이 로컬 Hugging Face 캐시에 없으면 다운로드한다. 이 모델은 `rag-vllm` 설정의 모델 ID를 가져온 것이며, vLLM 서버 주소나 vLLM 전용 메모리 설정은 사용하지 않는다.

```bash
# 선택 사항: bitsandbytes 4bit 양자화 실습 의존성
uv sync --extra quantization
```

실제 의미 검색은 `BAAI/bge-m3` 임베딩을 사용한다. 임베딩 장치는 기본 CPU라 생성 모델의 GPU 메모리와 경합하지 않는다. 기존 Ollama/vLLM 엔드포인트와 비교하려면 `.env`에서 `LLM_BACKEND=openai-compatible`을 선택하고 `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`을 설정한다. 검색만 실험하려면 `LLM_BACKEND=none`으로 둔다.

API를 실행한다.

```bash
# Transformers가 GPU를 쓸 때 rag-vllm의 vLLM 모델은 먼저 중지한다.
(cd ../rag-vllm && docker compose --profile vllm stop vllm)
uv run rag-ollama-api
```

다른 터미널에서 상태를 확인한다.

```bash
curl http://127.0.0.1:8000/health
```

아래 API 예시는 `RAG_LAB_API_KEY`를 설정하지 않은 로컬 기본값 기준이다. 키를 설정했다면 각 API 요청에 `-H 'X-API-Key: <실제 키>'`를 추가한다. `/health`는 키 없이 확인할 수 있다.

처음에는 BGE와 생성 모델 가중치를 내려받는다. `HF_DEVICE_MAP=auto`는 사용 가능한 GPU에 모델을 배치한다. `rag-vllm`의 vLLM이 실행 중이면 VRAM이 부족할 수 있으므로 둘 중 하나만 GPU 추론에 사용한다. CPU 실습에는 `HF_DEVICE_MAP=cpu`를 설정한다. 4bit 실습은 `uv sync --extra quantization` 후 `HF_QUANTIZATION=4bit`를 설정한다. 임베딩은 CPU 설정을 유지한다.

## 첫 실습: 등록, 검색, 답변

짧은 텍스트를 등록한다. `document_id`와 품질 점수, chunk 수를 기록해 둔다.

```bash
curl -X POST http://127.0.0.1:8000/documents/text \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "학습노트.md",
    "text": "연차휴가는 1년간 80퍼센트 이상 출근한 근로자에게 부여한다. 사용자는 근로자가 청구한 시기에 연차휴가를 주어야 한다.",
    "metadata": {"topic": "근로기준", "version": "학습용"}
  }'
```

먼저 검색만 실행해 `sources`의 근거 문장과 유사도 점수를 확인한다. 생성 답변과 비교하기 전에 검색 결과가 질문에 맞는지 살펴보면 RAG의 어느 단계에서 차이가 생겼는지 이해하기 쉽다.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"연차휴가는 어떤 조건으로 부여되나요?", "top_k":3, "use_llm":false}'
```

이후 `use_llm`을 `true`로 바꿔 Transformers 답변과 근거 번호를 확인한다. `top_k`, `min_quality_score`, `document_id`를 바꾸며 검색 범위를 비교해 볼 수 있다.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"연차휴가는 어떤 조건으로 부여되나요?", "top_k":3, "use_llm":true}'
```

## 고정 질문으로 모델 비교

PostgreSQL과 `rag-ollama-api`가 실행 중일 때 프로젝트 루트에서 고정 문서와 평가 질문을 실행한다. 평가 명령은 실행 중인 API를 호출해 fixture 문서를 멱등적으로 등록하고, 질문별 답변·검색 근거·설정·실제 모델 커밋·전체 지연시간·생성 토큰 속도를 `experiments/results/` 아래 JSONL 파일로 기록한다. 결과 파일은 Git에서 제외한다.

```bash
# 검색 근거만 비교
uv run rag-ollama-evaluate --retrieval-only

# 생성까지 포함해 실행 (첫 호출은 모델 다운로드·로드 시간이 포함됨)
uv run rag-ollama-evaluate
```

기본 질문 세트는 [`evals/questions.jsonl`](evals/questions.jsonl), 평가 문서는 [`evals/transformers-basics.md`](evals/transformers-basics.md)다. 질문마다 `expected_evidence`가 반환된 근거 안에 있는지 간단히 표시한다. `expected_answer`는 자동 채점이 아니라 답변을 사람이 비교할 참고 답이다. 다른 세트를 실행하려면 `--questions`, 다른 문서를 쓰려면 `--fixture`를 지정한다. `--top-k`로 질문 세트의 검색 개수를 덮어쓰고, 다른 주소의 API에는 `--api-url`을 지정한다.

`HF_REVISION`은 모델 ID에 맞는 Hugging Face 커밋 SHA로 설정한다. 현재 기본 Qwen 모델은 [이 커밋](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)에 고정되어 있고, 평가 파일에는 요청한 revision과 실제 로드된 revision을 모두 기록한다. 다른 모델로 바꾸면 그 저장소의 revision도 함께 바꾼다.

## 파일 등록과 품질 진단

텍스트, Markdown, CSV, JSON, HTML, PDF, DOCX 파일을 등록할 수 있다. 예를 들어:

```bash
curl -X POST http://127.0.0.1:8000/documents/file \
  -F 'file=@./sample.pdf' \
  -F 'metadata={"department":"품질","version":"1.0"}'
```

개별 텍스트를 등록하지 않고 기초 품질 진단만 요청할 수도 있다.

```bash
curl -X POST http://127.0.0.1:8000/quality/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"text":"진단할 문서 본문", "name":"sample.txt"}'
```

현재 진단은 빈 문서, 짧은 본문, 인코딩 대체 문자, 제어 문자, 반복 줄, 원천 이름 누락을 설명 가능한 규칙으로 점검하는 학습용 baseline이다. `score`와 `pass`/`warn`/`fail`은 휴리스틱 결과이며, 품질 표준 적합 판정이나 업무 승인 근거로 쓰면 안 된다. 실제 조직 기준은 버전이 있는 규칙과 검증 데이터로 별도 정의해야 한다.

`POST /quality/diagnose`에 등록 문서 ID를 주면 저장된 추출 텍스트를 현재 규칙으로 다시 계산한다. 응답의 `rule_set_version`을 비교할 수 있으며, 이 요청은 등록 시점의 품질 리포트를 수정하거나 별도 이력으로 저장하지 않는다. 여러 시점의 결과를 관리하는 진단 이력은 다음 리팩터링 후보이다.

스캔 PDF는 OCR이 필요하고 HWP/HWPX는 별도 어댑터가 필요하다. 파서는 파일 입력 25 MiB, PDF 2,000페이지, DOCX 압축 해제 100 MiB, 추출 텍스트 5,000,000자 상한을 둔다. 현재 지원하지 않는 형식은 명시적으로 오류를 반환한다.

## 설정 참고

`.env.example`을 복사해 `.env`에서 설정한다. 애플리케이션은 시작할 때 설정을 읽으므로 변경 후 API 프로세스를 다시 시작한다.

| 설정 | 기본값 | 설명 |
| --- | --- | --- |
| `DATABASE_URL` | Compose의 `localhost:55432` | PostgreSQL 연결 주소 |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | 로컬 전용 API 바인딩 |
| `EMBEDDING_PROVIDER` | `flag` | `flag`는 BGE 의미 임베딩, `hash`는 연결 확인용 가짜 임베딩 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-m3` / `1024` | 문서와 질문에 동일하게 쓸 임베딩 모델/차원 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1600` / `240` | 문자 단위 분할 크기와 겹침 |
| `RETRIEVAL_TOP_K` | `5` | 기본 검색 결과 수 |
| `MAX_UPLOAD_BYTES` | `20971520` | 업로드 기본 한도 20 MiB; 파서는 최대 25 MiB로 제한 |
| `LLM_BACKEND` | `transformers` | `transformers`, `openai-compatible`, `none` 중 선택 |
| `HF_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Hugging Face 모델 ID 또는 로컬 모델 경로 |
| `HF_REVISION` | 현재 Qwen 커밋 SHA | 모델 저장소의 커밋 SHA·태그·브랜치 |
| `HF_DEVICE_MAP` | `auto` | `auto`, `cpu`, `cuda:<index>` 중 선택 |
| `HF_DTYPE` | `auto` | 가중치 정밀도: `auto`, `float16`, `bfloat16`, `float32` |
| `HF_QUANTIZATION` | `none` | `none` 또는 선택 의존성 설치 후 `4bit` |
| `LLM_MAX_NEW_TOKENS` / `LLM_TEMPERATURE` | `512` / `0` | 생성 길이와 샘플링 온도 |
| `LLM_BASE_URL` / `LLM_MODEL` | 미설정 | `openai-compatible` 백엔드에서 Ollama/vLLM 연결에 사용 |
| `RAG_LAB_API_KEY` | 미설정 | 설정하면 `/health`를 제외한 API 요청에 `X-API-Key`가 필요 |

`EMBEDDING_PROVIDER=hash`는 연결 경로를 확인하기 위한 결정적 해시 벡터라서 의미 검색 품질을 보여주지 않는다. 실제 검색 학습에는 `flag`를 유지한다.

텍스트 등록·진단 입력은 최대 5,000,000자, 질문은 최대 5,000자다. 진단 응답의 `rule_set_version=baseline-text-v1`은 로컬 학습용 규칙의 버전이며 공식 기준 적합성을 뜻하지 않는다.

임베딩 모델이나 차원을 바꾸면 기존 문서의 벡터와 새 쿼리 벡터를 비교할 수 없다. 현재 자동 재임베딩/마이그레이션은 제공하지 않으므로, 모델을 바꾸기 전 데이터를 백업하고 새 빈 데이터베이스에서 문서를 다시 등록한다. 같은 차원의 다른 모델도 재등록이 필요하다.

## 로컬 접근과 보안

API 기본 주소는 `127.0.0.1`이고 Compose의 PostgreSQL 포트도 호스트 로컬에만 바인딩된다. 다른 장치에서 접근하도록 `API_HOST=0.0.0.0`으로 바꾸면 `.env`에 충분히 강한 `RAG_LAB_API_KEY`도 설정하고 네트워크 접근을 제한한다. API 키는 평문 HTTP 헤더이므로 원격 접근은 VPN 안에서 사용하거나 TLS 종료 프록시 뒤에 둔다. multipart 한도는 파싱 이후 라우트에서 검사하므로 원격 노출 시 앞단 프록시에도 요청 본문 크기 제한을 설정한다. API 키는 `/health`에는 적용되지 않는다. Transformers, Ollama, vLLM이 같은 GPU를 사용할 때는 추론 서버를 하나씩 실행하고, 전환할 때 이전 모델이 VRAM에 남아 있지 않도록 내린다.

## 다음 학습 과제

- 짧은 문서와 긴 문서를 넣고 `CHUNK_SIZE` 및 `CHUNK_OVERLAP`에 따라 반환되는 근거를 비교한다.
- `use_llm=false` 결과를 먼저 확인한 다음, 같은 검색 근거로 생성 답변이 달라지는지 살펴본다.
- `uv run rag-ollama-evaluate`로 고정 질문의 근거 적중률과 응답 시간을 모델별로 비교한다.
- `HF_DTYPE`과 `HF_QUANTIZATION`을 바꾸고 첫 생성 지연시간·메모리·답변 품질을 기록한다.
- 같은 질문과 검색 문맥을 Transformers와 OpenAI 호환 백엔드에 보내 결과를 비교한다.
- `quality.py`의 점수 규칙을 읽고 업무 기준에 맞는 새 규칙을 추가한 뒤, 규칙 버전과 검증 사례를 기록한다.
- PostgreSQL의 저장 레코드에서 원문, chunk, 메타데이터, 품질 리포트를 연결해 확인한다.

## API 목록

- `GET /health`: API 및 DB 상태
- `POST /documents/text`: 텍스트 등록
- `POST /documents/file`: 파일 추출 및 등록
- `GET /documents/{document_id}`: 등록 문서와 품질 요약
- `POST /query`: 벡터 검색 및 선택적 Transformers/OpenAI 호환 백엔드 답변
- `POST /quality/diagnose`: 단일 문서 또는 등록 문서의 품질 리포트

`POST /query` 응답의 `generation`에는 사용 백엔드·모델·revision, 생성 시간, 입출력 토큰 수와 생성 토큰 속도가 포함된다. Transformers 첫 호출의 `elapsed_ms`에는 모델 다운로드·로드 시간도 포함될 수 있다.
