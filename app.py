import json
import os
import time
import uuid
from datetime import datetime
import hashlib

import streamlit as st


# =========================
# 고정 프롬프트 (사용자 제공 단일본)
# =========================
BASE_PROMPT = """블로그 글 자동 생성 통합 프롬프트 (단일본)

당신은 10년 이상 경력의 전문 블로그 콘텐츠 작가 AI입니다.
네이버 블로그와 티스토리 블로그의 플랫폼 특성을 정확히 이해하고 있으며,
사람이 직접 작성한 것처럼 자연스럽고 신뢰감 있는 글을 작성합니다.

🔹 입력값

다음 정보를 기준으로 글을 작성하십시오.

blog_type: "네이버" | "티스토리" | "통합"

topic: 블로그 글 주제

keyword_list: 필수 포함 키워드 (없으면 자연스럽게 작성)

🔹 공통 필수 규칙 (절대 준수)

한글 기준 2,500자 이상 작성

띄어쓰기 제외

의미 없는 반복 금지

AI 느낌 제거

기계적 나열 금지

실제 경험을 이야기하듯 자연스럽게 전개

SEO 구조 준수

제목, 소제목 명확

키워드는 문맥에 맞게 자연스럽게 분산

결과물만 출력

설명, 해설, 주석, 안내 문구 절대 출력 금지

🟢 네이버 블로그 작성 규칙

(blog_type = "네이버")

HTML 사용 금지 (순수 텍스트만 출력)

일상 대화체 + 정보형 혼합

아래 말투를 자연스럽게 섞어 사용

~했어요!

오늘은 ~하겠습니다!

아이코!

그래서 말인데요

사람의 감정과 상황 묘사 포함

출력 형식
제목

도입 문단 (경험 + 감정)

소제목 1
본문

소제목 2
본문

마무리 (개인적인 한마디)

🔵 티스토리 블로그 작성 규칙

(blog_type = "티스토리")

HTML 코드로 출력

<body> 태그 안의 내용만 작성

모든 문단에 강제 style 속성 적용

글자 수 기준:

2,500자 이상

띄어쓰기 제외

HTML 태그 제외한 순수 텍스트 기준

스타일 필수 요소

배경색

글자색

문단 여백

인용문 강조

출력 예시 구조
<body>
  <h2 style="color:#222;">제목</h2>

  <p style="background:#f8f9fa;color:#333;padding:16px;border-radius:8px;">
    본문
  </p>

  <blockquote style="background:#222;color:#fff;padding:16px;">
    핵심 인용문
  </blockquote>
</body>


전문적이고 신뢰감 있는 문체

문단별로 정보가 명확히 구분되도록 작성

🟣 통합 모드 작성 규칙

(blog_type = "통합")

네이버 블로그 버전 + 티스토리 블로그 버전을 모두 생성

출력 순서와 구분 문구는 반드시 아래와 같이 유지

출력 형식 (변경 금지)
===== 네이버 블로그 버전 =====
네이버 블로그 글 전체

===== 티스토리 블로그 버전 =====
<body>
티스토리 HTML 전체
</body>

🚫 절대 금지

글자 수 설명 출력 금지

“AI가 작성했습니다” 관련 문구 금지

마크다운 사용 금지

불필요한 안내 문장 출력 금지

✅ 최종 지시

위 모든 규칙을 완벽히 준수하여
입력된 topic에 대한 블로그 글을 즉시 생성하라.
"""

HISTORY_DIR = ".data"
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")


def ensure_history_dir() -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)


def load_history() -> list[dict]:
    ensure_history_dir()
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    ensure_history_dir()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def build_prompt(blog_type: str, topic: str, keywords: str) -> str:
    keywords_value = keywords.strip() if (keywords or "").strip() else "자연스럽게 작성"
    return (
        f"{BASE_PROMPT}\n\n"
        f"[입력 정보]\n"
        f"blog_type: {blog_type}\n"
        f"topic: {topic}\n"
        f"keyword_list: {keywords_value}\n"
    )


def count_chars_no_spaces(text: str) -> int:
    if not text:
        return 0
    return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))


def strip_html_tags_simple(html: str) -> str:
    # 간단한 순수 텍스트 길이 측정용(완전한 HTML 파서 아님)
    import re

    if not html:
        return ""
    no_script = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    no_style = re.sub(r"<style[\s\S]*?</style>", "", no_script, flags=re.IGNORECASE)
    no_tags = re.sub(r"<[^>]+>", "", no_style)
    return no_tags


def calc_effective_char_count(blog_type: str, result: str) -> int:
    if blog_type in ("티스토리", "통합"):
        pure = strip_html_tags_simple(result)
        return count_chars_no_spaces(pure)
    return count_chars_no_spaces(result)


def generate_with_chatgpt(prompt: str, api_key: str, model: str, temperature: float) -> str:
    from openai import OpenAI

    cleaned_key = "".join((api_key or "").strip().strip('"').strip("'").split())
    client = OpenAI(api_key=cleaned_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def generate_with_gemini(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: int = 90,
    max_retries: int = 2,
    max_output_tokens: int | None = None,
) -> str:
    import google.generativeai as genai

    cleaned_key = "".join((api_key or "").strip().strip('"').strip("'").split())
    genai.configure(api_key=cleaned_key)
    gmodel = genai.GenerativeModel(model_name=model)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            gen_cfg = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            try:
                resp = gmodel.generate_content(
                    prompt,
                    generation_config=gen_cfg,
                    request_options={"timeout": int(timeout_s)},
                )
            except TypeError:
                # 일부 버전/환경에서 request_options 지원이 다를 수 있어 fallback
                resp = gmodel.generate_content(
                    prompt,
                    generation_config=gen_cfg,
                )
            return getattr(resp, "text", "") or ""
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
                continue
            raise last_err


def list_gemini_models(api_key: str) -> list[str]:
    import google.generativeai as genai

    cleaned_key = "".join((api_key or "").strip().strip('"').strip("'").split())
    genai.configure(api_key=cleaned_key)

    names: list[str] = []
    for m in genai.list_models():
        supported = getattr(m, "supported_generation_methods", None) or getattr(m, "supported_methods", None) or []
        if "generateContent" not in supported:
            continue
        name = getattr(m, "name", "") or ""
        if name.startswith("models/"):
            name = name[len("models/") :]
        if name:
            names.append(name)
    # 중복 제거 + 보기 좋게 정렬
    names = sorted(set(names))
    return names


def fingerprint_key(key: str) -> str:
    k = "".join((key or "").strip().strip('"').strip("'").split())
    if not k:
        return ""
    return hashlib.sha256(k.encode("utf-8")).hexdigest()[:12]


def is_likely_google_api_key(key: str) -> bool:
    k = "".join((key or "").strip().strip('"').strip("'").split())
    # Google API Key는 보통 AIza로 시작하는 경우가 많음(절대 규칙은 아님)
    return len(k) >= 20 and k.startswith("AIza")


def render_gemini_key_help(error_text: str | None = None) -> None:
    st.error("Gemini API Key가 유효하지 않습니다(API_KEY_INVALID).")
    st.write(
        "- **Google AI Studio에서 발급한 Gemini API Key**인지 확인해주세요.\n"
        "- 다른 서비스 키(OpenAI 키 등)를 Gemini 칸에 넣으면 이 오류가 납니다.\n"
        "- Google Cloud Console에서 만든 API Key라면, 해당 프로젝트에서 **Generative Language API**가 사용 가능/허용 상태인지 확인해주세요.\n"
        "- 키에 **제한(HTTP referrer/IP 제한)**을 걸어두면 로컬 앱에서 실패할 수 있으니, 테스트 중에는 제한을 잠시 해제해보세요."
    )
    if error_text:
        with st.expander("원본 오류(참고)"):
            st.code(error_text)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="AI 블로그 자동 작성기", layout="wide")
st.title("AI 블로그 글 자동 생성기")

if "openai_key" not in st.session_state:
    st.session_state.openai_key = os.getenv("OPENAI_API_KEY", "")
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = os.getenv("GEMINI_API_KEY", "")
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "last_generation" not in st.session_state:
    st.session_state.last_generation = None
if "gemini_models" not in st.session_state:
    st.session_state.gemini_models = []
if "gemini_key_fp" not in st.session_state:
    st.session_state.gemini_key_fp = ""
if "gemini_models_attempted" not in st.session_state:
    st.session_state.gemini_models_attempted = False

tab_generate, tab_history = st.tabs(["생성", "기록"])

with tab_generate:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        model_type = st.selectbox("AI 모델 선택", ["ChatGPT", "Gemini"])
        blog_type = st.selectbox("블로그 유형", ["네이버", "티스토리", "통합"])
        topic = st.text_input("블로그 주제", placeholder="예) 제주도 2박3일 여행 코스 후기")
        keywords = st.text_input("필수 키워드 (쉼표로 구분, 선택사항)", placeholder="예) 제주도, 렌터카, 맛집, 숙소")

        temperature = st.slider("창의성(temperature)", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        min_chars = st.number_input("최소 글자 수(띄어쓰기 제외) 경고 기준", min_value=0, value=2500, step=100)

        # 변수 초기화 (안전성을 위해 기본값 설정)
        api_key = ""
        selected_model_name = ""
        gemini_timeout_s = 90
        gemini_max_output_tokens: int | None = None
        debug_errors = False
        
        if model_type == "ChatGPT":
            openai_model = st.selectbox("ChatGPT 모델", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"])
            api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_key)
            st.session_state.openai_key = api_key
            selected_model_name = openai_model
        else:
            fallback_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "직접 입력(사용자 지정)"]

            api_key_raw = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_key)
            if any(ch.isspace() for ch in (api_key_raw or "")):
                st.warning("Gemini API Key에 공백/줄바꿈이 포함되어 있습니다. 자동으로 제거해서 사용합니다.")
            api_key = "".join((api_key_raw or "").strip().strip('"').strip("'").split())
            st.session_state.gemini_key = api_key
            if api_key and not is_likely_google_api_key(api_key):
                st.warning("입력한 키 형식이 Google API Key(AIza...)와 달라 보입니다. (OpenAI 키 등 다른 키를 넣지 않았는지 확인)")

            # 키가 바뀌면 모델 캐시 초기화
            new_fp = fingerprint_key(api_key)
            if new_fp != st.session_state.gemini_key_fp:
                st.session_state.gemini_key_fp = new_fp
                st.session_state.gemini_models = []
                st.session_state.gemini_models_attempted = False

            cbtn1, cbtn2 = st.columns([1, 3])
            with cbtn1:
                load_models_btn = st.button("모델 목록 불러오기", use_container_width=True)
            with cbtn2:
                st.caption("계정/리전에 따라 사용 가능한 모델이 다릅니다. 버튼을 누르면 generateContent 지원 모델만 불러옵니다.")

            # 키가 입력돼 있고 아직 시도 안 했으면 1회 자동 조회(사용자 UX 개선)
            if api_key and not st.session_state.gemini_models and not st.session_state.gemini_models_attempted:
                st.session_state.gemini_models_attempted = True
                with st.spinner("Gemini 모델 목록을 자동으로 확인 중..."):
                    try:
                        st.session_state.gemini_models = list_gemini_models(api_key)
                    except Exception as e:
                        # 자동 조회 실패는 기본적으로 조용히 넘기되, 키가 명백히 invalid면 안내
                        if "API_KEY_INVALID" in str(e):
                            render_gemini_key_help(str(e))

            if load_models_btn:
                if not api_key:
                    st.warning("Gemini API Key를 먼저 입력해주세요.")
                else:
                    with st.spinner("Gemini 모델 목록을 불러오는 중..."):
                        try:
                            st.session_state.gemini_models = list_gemini_models(api_key)
                            if not st.session_state.gemini_models:
                                st.warning("불러온 모델이 없습니다. API Key/권한/네트워크를 확인해주세요.")
                        except Exception as e:
                            if "API_KEY_INVALID" in str(e):
                                render_gemini_key_help(str(e))
                            else:
                                st.error(f"모델 목록 조회 실패: {e}")

            if st.session_state.gemini_models:
                model_options = st.session_state.gemini_models + ["직접 입력(사용자 지정)"]
                # 추천(Flash) 안내
                flash_candidates = [m for m in st.session_state.gemini_models if "flash" in m.lower()]
                if flash_candidates:
                    st.caption(f"추천(Flash): {', '.join(flash_candidates[:5])}" + (" ..." if len(flash_candidates) > 5 else ""))
            else:
                model_options = fallback_models

            gemini_model = st.selectbox("Gemini 모델", model_options)

            if gemini_model == "직접 입력(사용자 지정)":
                custom_model = st.text_input("사용할 모델명", placeholder="예) gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash")
                selected_model_name = (custom_model or "").strip()
            else:
                selected_model_name = gemini_model

            st.caption("팁: 무료 키/쿼터에서는 `gemini-1.5-flash`가 가장 안정적인 편입니다. 일부 최신 모델명은 SDK(v1beta)에서 미지원일 수 있어요.")
            gemini_timeout_s = st.slider("Gemini 요청 타임아웃(초)", 30, 240, 90, step=10)
            gemini_max_output_tokens = st.number_input("Gemini 최대 출력 토큰(선택)", min_value=0, value=0, step=256)
            gemini_max_output_tokens = None if int(gemini_max_output_tokens) <= 0 else int(gemini_max_output_tokens)
            debug_errors = st.checkbox("오류 상세 보기(디버그)", value=False)

        generate_btn = st.button("글 생성하기", type="primary")

    with col_right:
        st.subheader("미리보기")
        status_ph = st.empty()
        preview_ph = st.empty()
        actions_ph = st.empty()


def make_filename(blog_type_value: str, topic_value: str, ext: str) -> str:
    safe_topic = "".join(ch for ch in (topic_value or "").strip()[:30] if ch.isalnum() or ch in (" ", "_", "-")).strip()
    safe_topic = safe_topic.replace(" ", "_") if safe_topic else "blog"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{blog_type_value}_{safe_topic}_{ts}.{ext}"


with tab_generate:
    if generate_btn:
        # 입력 검증
        validation_errors = []
        if not api_key:
            validation_errors.append("API Key")
        if not topic.strip():
            validation_errors.append("주제")
        if model_type == "Gemini" and (not selected_model_name or not selected_model_name.strip()):
            validation_errors.append("Gemini 모델명")
        
        if validation_errors:
            status_ph.warning(f"다음 항목을 입력해주세요: {', '.join(validation_errors)}")
        else:
            final_prompt = build_prompt(blog_type=blog_type, topic=topic.strip(), keywords=keywords)
            with st.spinner("글을 생성 중입니다..."):
                try:
                    if model_type == "ChatGPT":
                        result = generate_with_chatgpt(
                            prompt=final_prompt,
                            api_key=api_key,
                            model=selected_model_name,
                            temperature=temperature,
                        )
                    else:
                        result = generate_with_gemini(
                            prompt=final_prompt,
                            api_key=api_key,
                            model=selected_model_name,
                            temperature=temperature,
                            timeout_s=int(gemini_timeout_s),
                            max_retries=2,
                            max_output_tokens=gemini_max_output_tokens,
                        )

                    st.session_state.last_result = result
                    status_ph.success("생성 완료!")

                    # 길이 체크 (요구사항 안내가 아니라, 앱 내부 품질 체크용)
                    n_chars = calc_effective_char_count(blog_type, result)
                    if min_chars and n_chars < int(min_chars):
                        st.warning(f"현재 글자 수(띄어쓰기 제외 추정): {n_chars} (기준 {int(min_chars)} 미만)")
                    else:
                        st.caption(f"글자 수(띄어쓰기 제외 추정): {n_chars}")

                    # 미리보기/저장용 상태 저장
                    st.session_state.last_generation = {
                        "blog_type": blog_type,
                        "topic": topic.strip(),
                        "result": result,
                        "char_count_no_spaces_est": int(n_chars),
                    }

                    # 기록 저장
                    entry = {
                        "id": str(uuid.uuid4()),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "blog_type": blog_type,
                        "topic": topic.strip(),
                        "keywords": (keywords or "").strip(),
                        "model_type": model_type,
                        "model": selected_model_name,
                        "temperature": float(temperature),
                        "char_count_no_spaces_est": int(n_chars),
                        "result": result,
                    }
                    st.session_state.history.insert(0, entry)
                    save_history(st.session_state.history)

                except Exception as e:
                    status_ph.error(f"오류 발생: {e}")
                    if model_type == "Gemini" and debug_errors:
                        st.exception(e)
                    if model_type == "Gemini":
                        msg = str(e)
                        if "API_KEY_INVALID" in msg:
                            render_gemini_key_help(msg)
                            st.stop()
                        if ("is not found for API version" in msg) or ("not found for API version" in msg) or ("is not found" in msg):
                            if api_key:
                                with st.spinner("사용 가능한 Gemini 모델을 다시 확인 중..."):
                                    try:
                                        st.session_state.gemini_models = list_gemini_models(api_key)
                                        if st.session_state.gemini_models:
                                            st.info("현재 API Key로 사용 가능한 모델 목록을 갱신했습니다. Gemini 모델 드롭다운에서 선택 후 다시 시도해보세요.")
                                        else:
                                            st.warning("모델 목록이 비어있습니다. 이 키로는 generateContent 모델 접근이 불가할 수 있습니다.")
                                    except Exception as e2:
                                        st.warning(f"모델 목록 재조회 실패: {e2}")

    # 미리보기는 항상 "저장"보다 먼저 보이도록, 생성 로직 이후에 렌더링
    lg = st.session_state.last_generation
    if lg and lg.get("result"):
        preview_ph.text_area("생성된 블로그 글", lg.get("result", ""), height=650)

        bt = lg.get("blog_type", "네이버")
        tp = lg.get("topic", "blog")
        if bt == "티스토리":
            ext = "html"
            mime = "text/html"
        else:
            ext = "txt"
            mime = "text/plain"

        actions_ph.download_button(
            "파일로 저장",
            data=lg.get("result", ""),
            file_name=make_filename(bt, tp, ext),
            mime=mime,
            use_container_width=True,
        )
    else:
        preview_ph.info("아직 생성된 글이 없습니다. 왼쪽에서 주제를 입력하고 생성해보세요.")

with tab_history:
    st.subheader("기록")
    st.caption("생성한 글은 자동 저장되며, 앱을 껐다 켜도 그대로 남습니다.")

    history_items: list[dict] = st.session_state.history or []

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("검색(주제/키워드/모델/유형)", placeholder="예) 제주도, 티스토리, gpt-4o-mini")
    with c2:
        filter_blog = st.selectbox("유형 필터", ["전체", "네이버", "티스토리", "통합"])
    with c3:
        filter_model_type = st.selectbox("모델 필터", ["전체", "ChatGPT", "Gemini"])

    def matches(item: dict) -> bool:
        if filter_blog != "전체" and item.get("blog_type") != filter_blog:
            return False
        if filter_model_type != "전체" and item.get("model_type") != filter_model_type:
            return False
        if not (q or "").strip():
            return True
        needle = q.strip().lower()
        hay = " | ".join(
            [
                str(item.get("topic", "")),
                str(item.get("keywords", "")),
                str(item.get("model_type", "")),
                str(item.get("model", "")),
                str(item.get("blog_type", "")),
                str(item.get("created_at", "")),
            ]
        ).lower()
        return needle in hay

    filtered = [it for it in history_items if matches(it)]

    if not filtered:
        st.info("저장된 기록이 없습니다.")
    else:
        labels = []
        id_to_item = {}
        for it in filtered:
            label = f"[{it.get('created_at','')}] {it.get('topic','(제목없음)')} · {it.get('blog_type','')} · {it.get('model_type','')}/{it.get('model','')}"
            labels.append(label)
            id_to_item[label] = it

        selected_label = st.selectbox("기록 선택", labels)
        selected = id_to_item.get(selected_label)

        if selected:
            meta_left, meta_right = st.columns([2, 1])
            with meta_left:
                st.write(
                    f"- 생성시간: {selected.get('created_at','')}\n"
                    f"- 블로그 유형: {selected.get('blog_type','')}\n"
                    f"- 모델: {selected.get('model_type','')}/{selected.get('model','')}\n"
                    f"- 주제: {selected.get('topic','')}\n"
                    f"- 키워드: {selected.get('keywords','')}\n"
                    f"- 글자 수(띄어쓰기 제외 추정): {selected.get('char_count_no_spaces_est','')}"
                )
            with meta_right:
                del_one = st.button("이 기록 삭제", type="secondary", use_container_width=True)
                clear_all = st.button("전체 기록 삭제", type="secondary", use_container_width=True)

            if del_one:
                st.session_state.history = [it for it in st.session_state.history if it.get("id") != selected.get("id")]
                save_history(st.session_state.history)
                st.rerun()
            if clear_all:
                st.session_state.history = []
                save_history(st.session_state.history)
                st.rerun()

            st.text_area("본문", selected.get("result", ""), height=650)

            # 기록 다운로드
            bt = selected.get("blog_type", "네이버")
            if bt == "티스토리":
                ext = "html"
                mime = "text/html"
            else:
                ext = "txt"
                mime = "text/plain"

            st.download_button(
                "선택한 기록 파일로 저장",
                data=selected.get("result", ""),
                file_name=make_filename(bt, selected.get("topic", "blog"), ext),
                mime=mime,
                use_container_width=True,
            )

        st.download_button(
            "전체 기록(JSON) 내보내기",
            data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

