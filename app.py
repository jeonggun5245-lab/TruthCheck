from __future__ import annotations

import json
import re
import time
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup

from core import (
    SCORE_LABELS, VALIDITY_LABELS, SCORE_KEYS, VALIDITY_KEYS,
    score_index, validate_public_http_url, parse_json_response,
    clean_source_ids, clamp_score_map,
)

st.set_page_config(page_title="TruthCheck AI", page_icon="🔎", layout="wide")
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

st.title("🔎 TruthCheck AI")
st.caption("기사 입력 → 근거 자동 수집 → AI 직접 분석 → 결과 보고서")
st.info("이 버전은 PC에서 로컬 AI 모델을 실행합니다. 외부 AI 유료 API는 사용하지 않습니다. 모델이 판단을 보조하지만, 근거가 부족한 주장은 자동으로 '평가 보류' 처리해야 합니다.")


def ollama_status() -> tuple[bool, list[str], str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        models = [x.get("name", "") for x in r.json().get("models", [])]
        return True, models, ""
    except Exception as exc:
        return False, [], str(exc)


def ollama_chat(model: str, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_ctx": 16384},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=360)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    except requests.RequestException as exc:
        raise RuntimeError(f"로컬 AI 호출에 실패했습니다: {exc}") from exc


def extract_article(raw: str) -> dict:
    raw = raw.strip()
    if not raw:
        raise ValueError("기사 URL 또는 본문을 입력해 주세요.")
    if re.match(r"^https?://", raw, flags=re.I):
        ok, reason = validate_public_http_url(raw)
        if not ok:
            raise ValueError(reason)
        r = requests.get(raw, headers={"User-Agent": "Mozilla/5.0 TruthCheck educational research"}, timeout=20)
        r.raise_for_status()
        final_url = r.url
        ok, reason = validate_public_http_url(final_url)
        if not ok:
            raise ValueError(reason)
        html = r.text
        metadata = trafilatura.extract_metadata(html)
        body = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True) or ""
        soup = BeautifulSoup(html, "html.parser")
        title = metadata.title if metadata and metadata.title else ""
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        if len(body.strip()) < 120:
            raise ValueError("기사 본문을 충분히 추출하지 못했습니다. 유료벽·로그인·사이트 차단일 수 있으니 본문을 직접 붙여 넣어 주세요.")
        return {"title": title, "url": final_url, "body": body.strip()[:24000]}
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    return {"title": lines[0][:200] if lines else "제목 미입력", "url": "", "body": raw[:24000]}


def gdelt_search(query: str, maxrecords: int = 5) -> list[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "timespan": "3m",
        "sort": "DateDesc",
    }
    r = requests.get(GDELT_URL, params=params, timeout=25, headers={"User-Agent": "TruthCheck educational project"})
    r.raise_for_status()
    data = r.json()
    articles = data.get("articles", []) if isinstance(data, dict) else []
    results = []
    for item in articles:
        url = item.get("url", "")
        if not url.startswith("https://") and not url.startswith("http://"):
            continue
        results.append({
            "title": item.get("title", "제목 없음"),
            "url": url,
            "domain": item.get("domain", "출처 도메인 미상"),
            "date": item.get("seendate", "날짜 미상"),
            "language": item.get("language", ""),
            "country": item.get("sourcecountry", ""),
            "tone": item.get("tone", None),
        })
    return results


def fetch_source_text(url: str) -> str:
    ok, _ = validate_public_http_url(url)
    if not ok:
        return ""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 TruthCheck educational research"}, timeout=12)
        r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=True, favor_precision=True) or ""
        return text.strip()[:4500]
    except Exception:
        return ""


def call_analysis(article: dict, model: str, progress) -> dict:
    claim_prompt = f"""너는 뉴스 기사 분석 시스템의 1단계 모듈이다. 다음 기사를 읽고 검증할 만한 핵심 '사실 주장'을 최대 4개 추출하라. 의견·평가·전망은 사실 주장에 섞지 말라. 각 주장마다 GDELT 뉴스 검색에 쓸 짧은 검색어(한국어 또는 고유명사 포함)를 하나 제안하라. 기사에 없는 내용을 만들지 말라. JSON 객체만 출력하라.
스키마: {{"claims":[{{"claim":"기사의 구체적 사실 주장","type":"사실 주장|의견·평가|전망·예측|해석","search_query":"짧은 검색어"}}],"companies":["기업명"],"article_summary":"기사 요약"}}
기사 제목: {article['title']}
기사 URL: {article['url'] or '본문 직접 입력'}
기사 본문: {article['body'][:16000]}
"""
    progress("① 기사에서 핵심 사실 주장을 추출하고 있어요.")
    first_raw = ollama_chat(model, [{"role": "user", "content": claim_prompt}])
    first = parse_json_response(first_raw)
    claims = first.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    claims = claims[:4]

    all_sources: list[dict] = []
    seen_urls = {article.get("url", "")}
    factual_queries = []
    for item in claims:
        if not isinstance(item, dict) or item.get("type") not in ("사실 주장", "해석"):
            continue
        q = str(item.get("search_query", "")).strip()[:180]
        if q and q not in factual_queries:
            factual_queries.append(q)
    for idx, query in enumerate(factual_queries[:3]):
        progress(f"② 관련 근거를 자동 검색하고 있어요. ({idx + 1}/{min(len(factual_queries), 3)})")
        try:
            found = gdelt_search(query, maxrecords=5)
        except Exception:
            found = []
        for item in found:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            all_sources.append(item)
        if idx < min(len(factual_queries), 3) - 1:
            time.sleep(5.2)  # GDELT public endpoint rate-limit courtesy

    # Prefer different domains, while retaining at most 8 source records.
    picked = []
    domains = set()
    for item in all_sources:
        domain = item.get("domain", "")
        if domain and domain not in domains:
            domains.add(domain)
            picked.append(item)
        if len(picked) >= 8:
            break
    for idx, item in enumerate(picked, 1):
        item["id"] = f"S{idx}"
        item["text"] = fetch_source_text(item["url"])

    evidence_block = "\n\n".join(
        f"[{s['id']}] 제목: {s['title']}\n도메인: {s['domain']} / 게시·수집 시각: {s['date']}\nURL: {s['url']}\n본문 발췌: {s['text'][:3500] or '본문 자동 추출 실패. 제목과 메타정보만 확보됨.'}"
        for s in picked
    ) or "검색 결과를 확보하지 못함. 검증 근거 없음."

    progress("③ 수집한 자료와 기사를 대조해 판단하고 있어요.")
    final_prompt = f"""너는 뉴스·기업 정보 검증 보조 시스템이다. 기사 내용을 실제로 검토하고, 검색으로 수집된 자료에 근거해 직접 분석 결과를 작성하라. 결론을 만들어 내지 말고 근거가 부족하면 반드시 '평가 보류' 또는 '근거 부족'이라고 하라.
중요 규칙:
- 부정적인 기사라는 이유만으로 허위·편향이라고 판단하지 말 것.
- 뉴스 기사 검색 결과가 원자료·공식문서와 같다고 가정하지 말 것.
- 서로 다른 도메인이라고 독립적 보도임이 자동 입증되는 것은 아님. 원출처·전재 여부가 불명확하면 독립성 확인 불가.
- 제공된 자료가 뒷받침하지 않는 숫자·사실·URL을 만들지 말 것.
- 출처를 언급할 때 제공된 S1~S8 ID만 사용. source_ids에 존재하지 않는 ID를 쓰지 말 것.
- 주가·시가총액 변화와 기업의 실제 현금/재무 손실을 분리할 것. 주가 하락만으로 기사와 손실의 인과관계를 단정하지 말 것.
- 평가 점수는 각 항목 1.0(충분히 확인), 0.5(일부 확인), 0.0(근거상 미충족), null(자료 부족으로 평가 불가) 중 하나. 점수는 진실일 확률이 아님.
- 출처가 부족하거나 반대 자료가 없으면 점수를 억지로 높이지 말고 null로 둘 것.

다음 JSON 스키마만 출력하라.
{{
 "summary":"기사 요약",
 "claims":[{{"claim":"주장","type":"사실 주장|의견·평가|전망·예측|해석","verdict":"확인|일부 확인·맥락 누락|근거 부족|허위·반박됨|평가 보류|의견·전망","reason":"판정 근거 및 한계","source_ids":["S1"]}}],
 "reliability":{{"source_transparency":null,"evidence_credibility":null,"independent_corroboration":null,"factual_accuracy":null,"timeliness_context":null}},
 "validity":{{"logic":null,"headline_match":null,"context":null,"counterevidence":null,"fact_opinion_distinction":null}},
 "reliability_notes":{{"source_transparency":"근거","evidence_credibility":"근거","independent_corroboration":"근거","factual_accuracy":"근거","timeliness_context":"근거"}},
 "validity_notes":{{"logic":"근거","headline_match":"근거","context":"근거","counterevidence":"근거","fact_opinion_distinction":"근거"}},
 "tone":"긍정|부정|중립|복합|판단 보류",
 "bias_signals":["선택적 정보 제시|감정적·선동적 표현|중요한 맥락 누락|출처 편중|과도한 일반화|사실과 의견의 혼합"],
 "bias_explanation":"편향 신호의 근거. 의도 단정 금지.",
 "companies":["기업명"],
 "audience_impact":"소비자·투자자에게 미칠 수 있는 영향",
 "corporate_impact":"기업에 미칠 수 있는 영향",
 "observed_market_change":"자료에서 직접 확인된 주가·시가총액 변화 또는 미확인",
 "documented_financial_impact":"재무제표·공식자료에서 확인된 실제 영향 또는 미확인",
 "causality_assessment":"해당 정보와 피해의 인과관계 수준 및 한계",
 "limitations":["자료의 한계"]
}}

[기사]
제목: {article['title']}
URL: {article['url'] or '본문 직접 입력'}
본문: {article['body'][:15000]}

[자동 검색 자료]
{evidence_block}
"""
    final_raw = ollama_chat(model, [{"role": "user", "content": final_prompt}], temperature=0.05)
    final = parse_json_response(final_raw)
    valid_ids = {s["id"] for s in picked}
    for item in final.get("claims", []) if isinstance(final.get("claims"), list) else []:
        if isinstance(item, dict):
            item["source_ids"] = clean_source_ids(item.get("source_ids"), valid_ids)
    final["reliability"] = clamp_score_map(final.get("reliability"), SCORE_KEYS)
    final["validity"] = clamp_score_map(final.get("validity"), VALIDITY_KEYS)
    final["reliability_score"] = score_index(final["reliability"], SCORE_KEYS)
    final["validity_score"] = score_index(final["validity"], VALIDITY_KEYS)
    final["sources"] = picked
    final["extracted_claims"] = claims
    final["companies"] = final.get("companies") or first.get("companies", [])
    return final


def render_score(title: str, result: dict, notes: dict, labels: dict):
    st.subheader(title)
    if result.get("score") is None:
        st.warning(f"평가 보류 — 평가 가능한 항목 {result.get('evaluated', 0)}/{result.get('total', 5)}개")
    else:
        st.metric(title, f"{result['score']:.1f}/5.0")
        st.caption(f"평가 가능 항목: {result['evaluated']}/{result['total']}개")
    for key, label in labels.items():
        value = result.get("items", {}).get(key) if isinstance(result.get("items"), dict) else None
        note = notes.get(key, "근거 설명 없음") if isinstance(notes, dict) else "근거 설명 없음"
        st.write(f"**{label}** — {value if value is not None else '평가 불가'}")
        st.caption(str(note))


with st.sidebar:
    st.header("로컬 AI 연결")
    model = st.text_input("Ollama 모델 이름", value=DEFAULT_MODEL, help="기본 설치 안내 모델: qwen3:4b")
    running, installed_models, err = ollama_status()
    if running:
        st.success("Ollama 서버 연결됨")
        if installed_models:
            st.caption("설치된 모델: " + ", ".join(installed_models))
    else:
        st.error("Ollama 서버 연결 안 됨")
        st.caption("Ollama 앱을 PC에서 실행한 뒤 새로고침하세요.")
    st.caption("AI 모델은 PC에서 실행되므로 모델 다운로드와 실행 속도는 PC 사양에 영향을 받습니다.")

raw_input = st.text_area("뉴스 기사 URL 또는 본문", height=180, placeholder="기사 URL을 입력하거나, 자동 추출이 안 되면 본문을 붙여 넣으세요.")
if st.button("기사 자동 분석 시작", type="primary", use_container_width=True, disabled=not running):
    try:
        with st.spinner("기사 본문을 가져오는 중..."):
            article = extract_article(raw_input)
        st.session_state["tc_article"] = article
        progress_box = st.empty()
        def progress(msg: str):
            progress_box.info(msg)
        with st.spinner("로컬 AI가 분석하고 있습니다. 검색 요청과 모델 처리에 시간이 걸릴 수 있어요."):
            result = call_analysis(article, model.strip() or DEFAULT_MODEL, progress)
        st.session_state["tc_result"] = result
        st.success("자동 분석이 완료되었습니다.")
    except Exception as exc:
        st.error(str(exc))
        st.exception(exc) if st.checkbox("기술 세부 오류 보기", value=False) else None

if st.session_state.get("tc_result"):
    article = st.session_state["tc_article"]
    result = st.session_state["tc_result"]
    st.divider()
    st.header("분석 결과")
    st.subheader(article.get("title") or "제목 미확인")
    if article.get("url"):
        st.markdown(f"[원문 기사 열기]({article['url']})")
    st.write(result.get("summary", "요약을 생성하지 못했습니다."))

    tab1, tab2, tab3, tab4 = st.tabs(["주장별 검증", "점수 평가", "편향·영향", "참고 자료"])
    with tab1:
        claims = result.get("claims", [])
        if claims:
            for idx, c in enumerate(claims, 1):
                if not isinstance(c, dict):
                    continue
                st.markdown(f"**{idx}. {c.get('claim', '주장 미확인')}**")
                st.write(f"유형: {c.get('type', '미분류')}　|　판정: **{c.get('verdict', '평가 보류')}**")
                st.write(c.get("reason", "근거 설명 없음"))
                ids = c.get("source_ids", [])
                if ids:
                    for src in result.get("sources", []):
                        if src.get("id") in ids:
                            st.markdown(f"- [{src['id']} · {src['title']}]({src['url']}) — {src['domain']}")
                else:
                    st.caption("연결된 자동 검색 출처가 없습니다.")
                st.divider()
        else:
            st.warning("핵심 주장 검증 결과가 없습니다.")
    with tab2:
        rel = result.get("reliability_score", {})
        val = result.get("validity_score", {})
        left, right = st.columns(2)
        with left:
            st.subheader("신뢰성")
            if rel.get("score") is None:
                st.warning(f"평가 보류 ({rel.get('evaluated', 0)}/5개 항목 평가 가능)")
            else:
                st.metric("신뢰성 점수", f"{rel['score']:.1f}/5.0")
            for k, label in SCORE_LABELS.items():
                value = result.get("reliability", {}).get(k)
                note = result.get("reliability_notes", {}).get(k, "")
                st.write(f"**{label}:** {value if value is not None else '평가 불가'}")
                st.caption(note)
        with right:
            st.subheader("타당성·균형성")
            if val.get("score") is None:
                st.warning(f"평가 보류 ({val.get('evaluated', 0)}/5개 항목 평가 가능)")
            else:
                st.metric("타당성·균형성 점수", f"{val['score']:.1f}/5.0")
            for k, label in VALIDITY_LABELS.items():
                value = result.get("validity", {}).get(k)
                note = result.get("validity_notes", {}).get(k, "")
                st.write(f"**{label}:** {value if value is not None else '평가 불가'}")
                st.caption(note)
        st.caption("점수는 기사가 참일 확률이 아니라, 제시한 평가 기준을 충족하는 정도를 나타내는 점검 지수입니다.")
    with tab3:
        st.write(f"**기사 어조:** {result.get('tone', '판단 보류')}")
        st.write("**편향 가능성 지표:** " + (", ".join(result.get("bias_signals", [])) or "뚜렷한 지표 미확인"))
        st.write(result.get("bias_explanation", ""))
        st.divider()
        st.subheader("기업·정보 이용자 영향")
        st.write("**관련 기업:** " + (", ".join(result.get("companies", [])) or "자동 식별되지 않음"))
        st.write("**소비자·투자자 영향:** " + str(result.get("audience_impact", "자료 부족")))
        st.write("**기업 영향:** " + str(result.get("corporate_impact", "자료 부족")))
        st.write("**관찰된 주가·시가총액 변화:** " + str(result.get("observed_market_change", "확인되지 않음")))
        st.write("**공식적으로 확인된 재무적 영향:** " + str(result.get("documented_financial_impact", "확인되지 않음")))
        st.write("**인과관계 평가:** " + str(result.get("causality_assessment", "평가 보류")))
    with tab4:
        sources = result.get("sources", [])
        if sources:
            for src in sources:
                st.markdown(f"**{src['id']} · [{src['title']}]({src['url']})**")
                st.caption(f"{src['domain']} · {src['date']} · 본문 추출 {'성공' if src.get('text') else '실패(제목·메타정보만 확보)'}")
        else:
            st.warning("자동 검색으로 관련 기사를 찾지 못했습니다. 이 경우 사실 검증은 제한적입니다.")
        st.markdown("검색 데이터: [GDELT Project](https://www.gdeltproject.org/)")
    report = [
        "# TruthCheck AI 분석 보고서",
        f"- 분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 기사 제목: {article.get('title', '')}",
        f"- 기사 URL: {article.get('url', '')}",
        "", "> 본 분석은 로컬 AI와 공개 뉴스 검색 결과를 이용한 보조적 검토입니다. 최종적인 사실 판정은 원자료 확인이 필요합니다.",
        "", "## 요약", result.get("summary", ""), "", "## 핵심 주장별 검증"
    ]
    for c in result.get("claims", []):
        if isinstance(c, dict):
            report += [f"### {c.get('claim', '')}", f"- 유형: {c.get('type', '')}", f"- 판정: {c.get('verdict', '')}", f"- 근거: {c.get('reason', '')}", f"- 출처 ID: {', '.join(c.get('source_ids', []))}", ""]
    report += ["## 신뢰성", f"- 점수: {rel.get('score') if rel.get('score') is not None else '평가 보류'}", "## 타당성·균형성", f"- 점수: {val.get('score') if val.get('score') is not None else '평가 보류'}", "## 어조·편향", f"- 어조: {result.get('tone', '')}", f"- 지표: {', '.join(result.get('bias_signals', []))}", result.get('bias_explanation', ''), "## 기업·정보 이용자 영향", f"- 기업: {', '.join(result.get('companies', []))}", f"- 소비자·투자자: {result.get('audience_impact', '')}", f"- 기업 영향: {result.get('corporate_impact', '')}", f"- 시장 변화: {result.get('observed_market_change', '')}", f"- 재무 영향: {result.get('documented_financial_impact', '')}", f"- 인과관계: {result.get('causality_assessment', '')}", "## 검색 출처"]
    for src in result.get("sources", []):
        report.append(f"- [{src['id']}] {src['title']} — {src['url']} ({src['domain']}, {src['date']})")
    report += ["", "## 한계"] + [f"- {x}" for x in result.get("limitations", [])]
    report += ["- GDELT 검색은 최근 약 3개월의 보도 범위로 제한되므로 오래된 사건은 자료가 없을 수 있습니다.", "- 검색된 기사가 독립적인 원자료인지 전재 기사인지는 별도 확인이 필요합니다.", "- AI는 오답이나 누락을 만들 수 있습니다. 출처 원문과 공시·정부 자료를 추가로 확인하세요."]
    st.download_button("분석 보고서 다운로드 (.md)", "\n".join(report), file_name="TruthCheck_Report.md", mime="text/markdown")

st.divider()
st.caption("TruthCheck AI · 학습·탐구 목적의 정보 검토 도구. 투자 권유·법률 판단·확정적인 진위 판정을 제공하지 않습니다.")
