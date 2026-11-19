from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import trafilatura

from core import (
    RELIABILITY_ITEMS,
    VALIDITY_ITEMS,
    SCORE_OPTIONS,
    calculate_score,
    validate_http_url,
    build_analysis_prompt,
    build_markdown_report,
    format_score,
)

st.set_page_config(page_title="TruthCheck AI", page_icon="🔎", layout="wide")

st.title("🔎 TruthCheck AI")
st.caption("뉴스·기업 정보의 신뢰성과 타당성을 검토하는 무료 수동 AI 연동형 분석 도구")

with st.container(border=True):
    st.markdown("**이 버전은 외부 AI API를 사용하지 않습니다.**")
    st.write("기사 URL 또는 본문을 입력하면 본문 추출을 시도하고, 무료 AI 서비스에 넣을 분석 프롬프트를 만들어 줍니다. AI가 작성한 분석 결과를 다시 붙여 넣으면 평가 항목과 보고서를 정리할 수 있습니다.")
    st.caption("주의: 프로그램 자체가 AI의 답변을 자동으로 수신하거나 사실을 독립적으로 검증하지 않습니다. 출처 원문 확인은 사용자가 수행해야 합니다.")


def extract_article(user_input: str) -> tuple[dict, str | None]:
    raw = user_input.strip()
    if not raw:
        return {}, "기사 URL 또는 본문을 입력해 주세요."
    if re.match(r"^https?://", raw, flags=re.I):
        ok, reason = validate_http_url(raw)
        if not ok:
            return {}, reason
        try:
            response = requests.get(
                raw,
                headers={"User-Agent": "Mozilla/5.0 (compatible; TruthCheck/1.0; educational project)"},
                timeout=15,
                allow_redirects=True,
            )
            response.raise_for_status()
            final_url = response.url
            ok_final, reason_final = validate_http_url(final_url)
            if not ok_final:
                return {}, f"리디렉션된 주소를 가져올 수 없습니다: {reason_final}"
            html = response.text
            metadata = trafilatura.extract_metadata(html)
            body = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True)
            soup = BeautifulSoup(html, "html.parser")
            title = (metadata.title if metadata and metadata.title else "")
            if not title:
                og = soup.find("meta", attrs={"property": "og:title"})
                title = (og.get("content") if og else "") or (soup.title.get_text(" ", strip=True) if soup.title else "")
            if not body or len(body.strip()) < 120:
                body = soup.get_text("\n", strip=True)
            body = body.strip()[:30000]
            if len(body) < 120:
                return {}, "본문을 충분히 추출하지 못했습니다. 유료벽·로그인·사이트 차단일 수 있습니다. 기사 본문을 직접 붙여 넣어 주세요."
            return {"title": title, "url": final_url, "body": body}, None
        except requests.RequestException as exc:
            return {}, f"URL에서 본문을 가져오지 못했습니다. 기사 본문을 직접 붙여 넣어 주세요. (연결 오류: {type(exc).__name__})"
        except Exception as exc:
            return {}, f"본문 추출 중 문제가 발생했습니다. 본문을 직접 붙여 넣어 주세요. (오류: {type(exc).__name__})"
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        title = lines[0][:200] if lines else ""
        return {"title": title, "url": "", "body": raw[:30000]}, None


if "article" not in st.session_state:
    st.session_state.article = None
if "ai_result" not in st.session_state:
    st.session_state.ai_result = ""

st.header("1. 기사 입력")
article_input = st.text_area(
    "기사 URL 또는 기사 본문",
    height=170,
    placeholder="https://example.com/news/... 또는 기사 본문을 붙여 넣으세요.",
    help="주소를 입력하면 본문 추출을 시도합니다. 실패하면 로그인·유료벽 등의 제한일 수 있으므로 본문을 붙여 넣어 주세요.",
)
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("기사 불러오기", type="primary", use_container_width=True):
        with st.spinner("기사 내용을 확인하고 있습니다..."):
            article, error = extract_article(article_input)
        if error:
            st.error(error)
        else:
            st.session_state.article = article
            st.success("기사 내용을 불러왔습니다.")
with col_b:
    st.caption("개인정보나 로그인 정보가 필요한 주소는 입력하지 마세요. 자동 본문 추출은 모든 사이트에서 작동하지 않습니다.")

article = st.session_state.article
if article:
    st.subheader("불러온 기사")
    st.text_input("기사 제목(필요하면 수정)", value=article.get("title", ""), key="edited_title")
    st.text_input("기사 URL", value=article.get("url", ""), key="edited_url")
    st.text_area("추출된 본문(필요하면 수정)", value=article.get("body", ""), height=220, key="edited_body")

    st.header("2. 무료 AI로 분석하기")
    target = st.radio(
        "분석 관점",
        ["기업", "소비자·투자자", "기업 및 소비자·투자자"],
        horizontal=True,
        key="target",
    )
    prompt = build_analysis_prompt(
        st.session_state.get("edited_title", article.get("title", "")),
        st.session_state.get("edited_url", article.get("url", "")),
        st.session_state.get("edited_body", article.get("body", "")),
        target,
    )
    with st.expander("AI 분석 프롬프트 보기", expanded=True):
        st.write("아래 프롬프트를 복사해 ChatGPT·Gemini·Perplexity 등에 붙여 넣으세요. 무료 이용량과 웹 검색 가능 여부는 서비스별로 다를 수 있습니다.")
        st.code(prompt, language="text")
    st.info("① 위 프롬프트 복사 → ② 무료 AI에 붙여 넣기 → ③ AI의 분석 결과를 아래 칸에 붙여 넣기")
    st.session_state.ai_result = st.text_area(
        "외부 AI의 분석 결과 붙여 넣기",
        value=st.session_state.ai_result,
        height=220,
        placeholder="AI가 제시한 주장별 검증 결과, 근거, 출처 URL 등을 붙여 넣으세요.",
        key="ai_result_box",
    )

    st.header("3. 핵심 주장별 검토 기록")
    st.write("AI 분석과 원자료를 확인한 뒤, 중요한 주장만 행별로 정리하세요. 출처 URL을 함께 남기는 것이 좋습니다.")
    default_claims = pd.DataFrame([
        {"핵심 주장": "", "검토 결과": "평가 보류", "근거 요약": "", "출처 URL": ""},
    ])
    if "claims_df" not in st.session_state:
        st.session_state.claims_df = default_claims
    edited_claims = st.data_editor(
        st.session_state.claims_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "핵심 주장": st.column_config.TextColumn("핵심 주장", width="large"),
            "검토 결과": st.column_config.SelectboxColumn("검토 결과", options=["확인", "일부 확인·맥락 누락", "근거 부족", "허위·반박됨", "의견·전망", "평가 보류"], width="medium"),
            "근거 요약": st.column_config.TextColumn("근거 요약", width="large"),
            "출처 URL": st.column_config.TextColumn("출처 URL", width="large"),
        },
        key="claims_editor",
    )
    st.session_state.claims_df = edited_claims

    st.header("4. 신뢰성·타당성 평가")
    st.caption("각 항목을 '충분히 확인됨(1.0)', '일부 확인됨(0.5)', '미충족(0.0)', '평가 불가' 중 하나로 직접 선택하세요. 평가 불가는 점수 계산에서 제외합니다.")
    reliability_values = {}
    validity_values = {}
    left, right = st.columns(2)
    with left:
        st.subheader("신뢰성")
        for i, (label, description) in enumerate(RELIABILITY_ITEMS):
            reliability_values[label] = st.selectbox(
                f"{i+1}. {label}", list(SCORE_OPTIONS.keys()), index=3,
                help=description, key=f"rel_{i}"
            )
    with right:
        st.subheader("타당성·균형성")
        for i, (label, description) in enumerate(VALIDITY_ITEMS):
            validity_values[label] = st.selectbox(
                f"{i+1}. {label}", list(SCORE_OPTIONS.keys()), index=3,
                help=description, key=f"val_{i}"
            )
    rel_score = calculate_score(list(reliability_values.values()))
    val_score = calculate_score(list(validity_values.values()))
    m1, m2 = st.columns(2)
    m1.metric("신뢰성 점수", format_score(rel_score))
    m2.metric("타당성·균형성 점수", format_score(val_score))
    st.caption("이 점수는 기사 진실 확률이 아니라 평가 기준에 따른 점검 지수입니다. 평가 가능한 항목이 3개 미만이면 평가 보류로 표시됩니다.")

    st.header("5. 어조와 편향 가능성")
    c1, c2 = st.columns(2)
    with c1:
        tone = st.selectbox("기사 어조", ["판단 보류", "긍정", "부정", "중립", "복합"], key="tone")
    with c2:
        bias_choices = ["선택적 정보 제시", "감정적·선동적 표현", "중요한 맥락 누락", "출처 편중", "과도한 일반화", "사실과 의견의 혼합"]
        bias = st.multiselect("관찰된 편향 가능성 지표", bias_choices, key="bias")
    bias_notes = st.text_area("편향 판단의 근거 및 유의점", key="bias_notes", height=90)
    st.caption("편향 지표가 있다고 해서 의도적인 조작이 입증되는 것은 아닙니다. 기사 어조 역시 사실성 점수와 별개로 판단합니다.")

    st.header("6. 기업 영향 분석(선택)")
    company_enabled = st.checkbox("이 기사는 특정 기업과 관련되어 있어 추가 분석하기", value=False)
    company = market_change = financial_impact = causality = company_notes = ""
    if company_enabled:
        company = st.text_input("기업명", key="company")
        market_change = st.text_area("관찰된 주가·시가총액 변화 (기간·수치·자료 출처 포함)", key="market_change", height=80)
        financial_impact = st.selectbox(
            "확인된 재무적 영향의 유형",
            ["자료 없음·확인 불가", "시가총액·주가 변화만 확인", "재무제표상 실제 손실 확인", "매출·영업이익 등 실적 변화 확인", "잠재적 영향만 제시됨"],
            key="financial_impact",
        )
        causality = st.selectbox(
            "기사·정보와 경제적 피해의 인과관계",
            ["평가 불가", "시간적 선후관계만 확인", "영향 가능성은 있으나 인과관계 미확인", "다른 요인까지 검토한 근거 있음", "인과관계를 뒷받침하는 공식 조사·자료 있음"],
            key="causality",
        )
        company_notes = st.text_area("영향 분석의 근거와 한계", key="company_notes", height=100)
        st.warning("주가 하락은 기업의 현금 손실과 다릅니다. 기사 게시 전후의 주가 변화만으로 해당 기사가 하락을 일으켰다고 단정하지 마세요.")

    st.header("7. 보고서 만들기")
    if st.button("분석 보고서 생성", type="primary"):
        claims_text_lines = []
        for _, row in edited_claims.iterrows():
            claim = str(row.get("핵심 주장", "") or "").strip()
            if claim:
                claims_text_lines.append(
                    f"- **주장:** {claim}\n  - 검토 결과: {row.get('검토 결과', '평가 보류')}\n  - 근거: {str(row.get('근거 요약', '') or '').strip()}\n  - 출처: {str(row.get('출처 URL', '') or '').strip()}"
                )
        report_data = {
            "title": st.session_state.get("edited_title", article.get("title", "")),
            "url": st.session_state.get("edited_url", article.get("url", "")),
            "target": target,
            "reliability": reliability_values,
            "validity": validity_values,
            "reliability_score": rel_score,
            "validity_score": val_score,
            "claims": "\n".join(claims_text_lines) or "검토 항목 없음",
            "tone": tone,
            "bias": bias,
            "bias_notes": bias_notes,
            "company_enabled": company_enabled,
            "company": company,
            "market_change": market_change,
            "financial_impact": financial_impact,
            "causality": causality,
            "company_notes": company_notes,
            "ai_result": st.session_state.get("ai_result_box", ""),
        }
        report = build_markdown_report(report_data)
        st.session_state.report = report
        st.success("보고서를 생성했습니다.")
    if st.session_state.get("report"):
        st.download_button(
            "Markdown 보고서 다운로드",
            data=st.session_state.report,
            file_name=f"TruthCheck_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

else:
    st.info("먼저 기사 URL 또는 본문을 입력하고 '기사 불러오기'를 눌러 주세요.")

with st.expander("이 프로그램의 평가 기준"):
    st.markdown("**신뢰성:** 출처의 투명성, 근거의 신뢰성, 독립적 교차 확인, 사실관계의 정확성, 시의성 및 맥락")
    st.markdown("**타당성·균형성:** 근거와 결론의 논리적 연결, 제목과 본문의 일치, 맥락의 충분성, 관련 반대 근거·당사자 입장, 사실과 의견의 구분")
    st.markdown("각 항목은 1.0 / 0.5 / 0.0 또는 평가 불가로 기록합니다. 평가 불가는 평균 계산에서 제외하며, 5개 중 3개 미만만 평가할 수 있으면 총점을 보류합니다.")
