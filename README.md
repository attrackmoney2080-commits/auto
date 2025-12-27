# AI 블로그 글 자동 생성기 (Streamlit)

ChatGPT(OpenAI) / Gemini(Google) 모델을 선택해서 **네이버/티스토리/통합** 블로그 글을 생성하는 Streamlit 앱입니다.  
프롬프트는 질문에서 제공한 **단일 통합 프롬프트**를 그대로 사용합니다.

## 실행 방법

### 1) 설치

```bash
pip install -r requirements.txt
```

### 2) 실행

```bash
streamlit run app.py
```

## 기록(History)

생성된 결과는 자동으로 로컬에 저장됩니다: `.data/history.json`  
앱의 **기록** 탭에서 이전에 만든 글을 검색/확인/다운로드/삭제할 수 있습니다.

## 환경변수(선택)

앱에서 키를 직접 입력해도 되고, 아래 환경변수를 쓰면 기본값으로 자동 채워집니다.

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`


