"""Core logic for TruthCheck's no-API prototype."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
import ipaddress
import socket


SCORE_OPTIONS = {
    "충분히 확인됨": 1.0,
    "일부 확인됨": 0.5,
    "확인되지 않거나 기준 미충족": 0.0,
    "평가 불가": None,
}

RELIABILITY_ITEMS = [
    ("출처의 투명성", "작성자·발행처·게시 시점 등 출처 정보를 확인할 수 있는가?"),
    ("근거의 신뢰성", "주장을 뒷받침하는 자료가 공식 문서·원자료·전문기관 등 신뢰할 만한 근거인가?"),
    ("독립적 교차 확인", "서로 독립적인 자료에서 핵심 사실을 확인할 수 있는가? 동일 보도자료의 반복 인용은 독립 확인으로 보지 않는다."),
    ("사실관계의 정확성", "주요 사실 주장들이 확인된 자료와 일치하는가?"),
    ("시의성 및 맥락", "자료의 작성 시점이 적절하고 필요한 시간적·상황적 맥락이 제시되어 있는가?"),
]

VALIDITY_ITEMS = [
    ("근거와 결론의 논리적 연결", "제시된 근거로 기사 속 결론을 뒷받침할 수 있는가?"),
    ("제목과 본문의 일치", "제목이 본문의 실제 내용과 과장 없이 부합하는가?"),
    ("맥락의 충분성", "독자가 사실을 오해하지 않도록 배경·조건·한계를 제시하는가?"),
    ("관련 반대 근거·당사자 입장", "판단에 필요한 반대 근거나 관련 당사자의 입장을 적절히 다루는가?"),
    ("사실과 의견의 구분", "확인된 사실, 해석, 의견, 전망을 구분해 제시하는가?"),
]


def calculate_score(values: list[str], minimum_evaluable: int = 3) -> dict[str, Any]:
    """Calculate a rubric index on a 0.0–5.0 scale, excluding unevaluable items."""
    scored = [SCORE_OPTIONS.get(value) for value in values]
    evaluable = [value for value in scored if value is not None]
    count = len(evaluable)
    if count < minimum_evaluable:
        return {"status": "평가 보류", "score": None, "evaluated": count, "total": len(values)}
    score = round((sum(evaluable) / count) * 5, 1)
    return {"status": "평가 완료", "score": score, "evaluated": count, "total": len(values)}


def validate_http_url(value: str) -> tuple[bool, str]:
    """Basic public HTTP(S) URL validation before attempting article retrieval."""
    try:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False, "http:// 또는 https://로 시작하는 완전한 URL을 입력해 주세요."
        host = parsed.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False, "로컬 네트워크 주소는 가져올 수 없습니다."
        try:
            ip = ipaddress.ip_address(host)
            if not ip.is_global:
                return False, "공개 인터넷 주소만 입력할 수 있습니다."
        except ValueError:
            # Hostnames are accepted; the app does not resolve arbitrary DNS names here.
            pass
        return True, ""
    except Exception:
        return False, "URL 형식을 확인해 주세요."


def build_analysis_prompt(title: str, source_url: str, article_text: str, target: str) -> str:
    article_text = article_text[:22000]
    return f"""너는 정보 검증을 돕는 조사 보조자다. 기사를 무조건 참/거짓으로 단정하지 말고, 확인 가능한 근거와 불확실성을 분리해 분석하라.

[분석 관점]
{target}

[기사 정보]
제목: {title or '제목 확인 불가'}
URL: {source_url or '사용자가 본문을 직접 입력함'}

[기사 본문]
{article_text}

[분석 지시]
1. 기사의 핵심 주장을 가능한 한 원문에 가깝게 문장 단위로 추출하라.
2. 각 주장을 다음 중 하나로 구분하라: 사실 주장 / 의견·평가 / 전망·예측 / 해석.
3. 사실 주장은 각각 확인하라. 공식 기관·공시·법원·연구기관·원자료를 우선하고, 독립적인 언론 보도도 보조적으로 활용하라.
4. 출처마다 발행 기관, 문서·기사 제목, 날짜, URL, 실제로 뒷받침하는 내용을 적어라. 출처를 만들어 내지 말고, 확인하지 못한 자료는 확인 불가라고 표시하라.
5. 여러 매체가 같은 보도자료를 반복 인용한 경우 독립적인 교차 확인으로 계산하지 말라.
6. 기사에 불리한 내용이 있다는 이유만으로 편향 또는 허위라고 판단하지 말라. 사실에 근거한 비판 보도는 신뢰할 수 있을 수 있다.
7. 제목과 본문의 차이, 누락된 맥락, 감정적·단정적 표현, 선택적 정보 제시 등 편향 가능성을 별도로 설명하라. 의도적인 조작 여부는 근거 없이 추정하지 말라.
8. 기업 관련 내용이라면 소비자·투자자에게 미칠 수 있는 영향과 기업에 미칠 수 있는 영향을 나누어 설명하라.
9. 주가·시가총액 변화와 매출·영업이익·순이익 등 실제 경영 실적을 구분하라. 주가 하락을 기업의 현금 손실로 표현하지 말라.
10. 특정 뉴스가 경제적 피해를 일으켰다는 인과관계는 시간적 선후관계만으로 단정하지 말라. 시장·산업·기업 자체의 다른 요인을 함께 검토하라.

[출력 형식]
A. 기사 요약 (3~5문장)
B. 핵심 주장 표: 주장 / 유형 / 판정(확인·일부 확인·근거 부족·반박됨·평가 보류) / 근거 / 출처 URL
C. 출처의 신뢰성과 독립성 검토
D. 제목·맥락·표현상의 편향 가능성
E. 기사 어조: 긍정 / 부정 / 중립 / 복합 (사실성 평가와 분리)
F. 기업 관련 영향(해당할 때만): 관찰된 시장 반응 / 확인된 재무상 영향 / 잠재적 영향 / 인과관계의 근거와 한계
G. 확인하지 못한 사항과 추가로 필요한 자료

마지막에 이 분석은 법률·투자 조언이나 확정적인 진위 판정이 아니라, 자료에 근거한 검토라는 점을 밝혀라."""


def build_markdown_report(data: dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# TruthCheck 분석 보고서",
        f"- 생성 시각: {now}",
        f"- 기사 제목: {data.get('title') or '미입력'}",
        f"- 기사 URL: {data.get('url') or '미입력'}",
        f"- 분석 관점: {data.get('target') or '미선택'}",
        "",
        "> 이 보고서는 사용자가 입력한 평가와 AI 보조 분석 내용을 정리한 자료다. 점수는 기사 진실 확률이 아니라 아래 평가 기준에 따른 점검 지수다.",
        "",
        "## 1. 신뢰성 평가",
    ]
    for label, _ in RELIABILITY_ITEMS:
        lines.append(f"- {label}: {data['reliability'].get(label, '평가 불가')}")
    lines.append(f"- **신뢰성 점수:** {format_score(data['reliability_score'])}")
    lines += ["", "## 2. 타당성·균형성 평가"]
    for label, _ in VALIDITY_ITEMS:
        lines.append(f"- {label}: {data['validity'].get(label, '평가 불가')}")
    lines.append(f"- **타당성·균형성 점수:** {format_score(data['validity_score'])}")
    lines += ["", "## 3. 핵심 주장별 검토", data.get("claims", "검토 항목 없음"), "", "## 4. 기사 어조 및 편향 지표"]
    lines.append(f"- 기사 어조: {data.get('tone', '평가 보류')}")
    lines.append(f"- 편향 가능성 지표: {', '.join(data.get('bias', [])) or '특별히 선택하지 않음'}")
    lines.append(data.get("bias_notes", ""))
    if data.get("company_enabled"):
        lines += ["", "## 5. 기업 관련 영향 분석"]
        for key, label in [
            ("company", "기업명"), ("market_change", "관찰된 주가·시가총액 변화"),
            ("financial_impact", "재무적 영향 구분"), ("causality", "기사와 피해의 인과관계 평가"),
            ("company_notes", "추가 근거 및 한계"),
        ]:
            lines.append(f"- **{label}:** {data.get(key) or '미입력'}")
    lines += ["", "## 6. 외부 AI 보조 분석 원문", data.get("ai_result", "미입력"), "", "## 7. 한계", "- 자동 URL 본문 추출은 사이트의 접근 제한·로그인·유료벽 등에 따라 실패할 수 있다.", "- AI가 제시한 출처와 수치는 사용자가 원문을 직접 확인해야 한다.", "- 평가 불가 항목은 0점으로 처리하지 않는다. 평가 가능한 항목이 3개 미만이면 점수를 산출하지 않는다.", "- 기사 게시 전후의 주가 변동만으로 인과관계를 확정할 수 없다."]
    return "\n".join(lines)


def format_score(result: dict[str, Any]) -> str:
    if not result or result.get("score") is None:
        return f"평가 보류 ({result.get('evaluated', 0)}/{result.get('total', 5)}개 평가 가능)"
    return f"{result['score']:.1f}/5.0 ({result['evaluated']}/{result['total']}개 평가)"
