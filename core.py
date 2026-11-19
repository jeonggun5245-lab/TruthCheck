from __future__ import annotations

import ipaddress
import json
import re
import socket
from typing import Any
from urllib.parse import urlparse

SCORE_LABELS = {
    "source_transparency": "출처의 투명성",
    "evidence_credibility": "근거의 신뢰성",
    "independent_corroboration": "독립적 교차 확인",
    "factual_accuracy": "사실관계의 정확성",
    "timeliness_context": "시의성 및 맥락",
}
VALIDITY_LABELS = {
    "logic": "근거와 결론의 논리적 연결",
    "headline_match": "제목과 본문의 일치",
    "context": "맥락의 충분성",
    "counterevidence": "관련 반대 근거·당사자 입장",
    "fact_opinion_distinction": "사실과 의견의 구분",
}
SCORE_KEYS = list(SCORE_LABELS)
VALIDITY_KEYS = list(VALIDITY_LABELS)


def score_index(items: dict[str, Any], keys: list[str], minimum: int = 3) -> dict[str, Any]:
    values: list[float] = []
    for key in keys:
        raw = items.get(key)
        if isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value in (0.0, 0.5, 1.0):
            values.append(value)
    if len(values) < minimum:
        return {"status": "평가 보류", "score": None, "evaluated": len(values), "total": len(keys)}
    return {"status": "평가 완료", "score": round(sum(values) / len(values) * 5, 1), "evaluated": len(values), "total": len(keys)}


def validate_public_http_url(value: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False, "http:// 또는 https://로 시작하는 주소를 입력해 주세요."
        host = parsed.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False, "로컬 주소는 가져올 수 없습니다."
        try:
            infos = socket.getaddrinfo(host, None)
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if not ip.is_global:
                    return False, "공개 인터넷 주소만 가져올 수 있습니다."
        except socket.gaierror:
            return False, "주소의 서버를 찾을 수 없습니다."
        except ValueError:
            return False, "주소 형식을 확인해 주세요."
        return True, ""
    except Exception:
        return False, "주소 형식을 확인해 주세요."


def parse_json_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("AI 응답에서 JSON 형식을 찾지 못했습니다.")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("AI 응답의 최상위 형식이 객체가 아닙니다.")
    return value


def clean_source_ids(ids: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(ids, list):
        return []
    return [str(x) for x in ids if str(x) in valid_ids]


def clamp_score_map(value: Any, keys: list[str]) -> dict[str, float | None]:
    source = value if isinstance(value, dict) else {}
    out: dict[str, float | None] = {}
    for key in keys:
        raw = source.get(key)
        if raw is None or raw == "평가 불가":
            out[key] = None
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            out[key] = None
            continue
        out[key] = num if num in (0.0, 0.5, 1.0) else None
    return out
