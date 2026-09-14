"""Deterministic, relation-preserving retrieval plans for visual VQA.

The question may describe the answer source (for example a value on a scale)
without identifying the video by itself.  This module turns that source into
one extra visual query while keeping the source attached to its device.  It
never expands a scale display into a generic screen, sign, or price board.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata


def fold_text(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def quantity_subject(question: str) -> str | None:
    """Extract the explicitly named entity in a quantity question."""
    match = re.search(r'\bluong\s+(.+?)\s+(?:duoc|can|su dung|la|bao nhieu)\b', fold_text(question))
    return match.group(1).strip() if match else None


_VISIBLE_TEXT = re.compile(
    r"\b(so hien thi|hien thi|man hinh|mat so|dong ho|display|screen|readout|shown|visible)\b"
)
_NUMERIC = re.compile(
    r"\b(con so|bao nhieu|gia tri|can nang|trong luong|nhiet do|toc do|"
    r"number|how much|how many|value|weight|temperature|speed)\b"
)
_LAST = re.compile(r"\b(cuoi cung|sau cung|ngay truoc|last|final|before the scene)\b")
_CHART = re.compile(r"\b(bieu do|do thi|chart|graph|plotted curve|axis)\b")
_MISSING_VISIBLE = re.compile(
    r"\b(khong (?:duoc )?(?:nhin thay|xuat hien)|khong thay|"
    r"not (?:seen|visible|shown)|never (?:seen|shown)|missing)\b"
)
_FIRST_SECONDS = re.compile(
    r"(?:(?P<vi>\d+(?:[.,]\d+)?)\s*giay\s*dau tien|"
    r"first\s*(?P<en>\d+(?:[.,]\d+)?)\s*seconds?)\b"
)
_VALUE_RANGE = re.compile(
    r"\b(?:tu|from)\s*(?P<start>\d+)\s*(?:-|–|—|den|toi|to)\s*(?P<end>\d+)\b"
)

# These are device nouns, not answer guesses.  More devices can be added
# without changing the routing policy or allowing unrelated generic screens.
_DEVICES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(can dien tu|cai can|can trong luong|weighing scale|electronic scale|digital scale|scale)\b"), "weighing scale"),
    (re.compile(r"\b(nhiet ke|thermometer)\b"), "thermometer"),
    (re.compile(r"\b(dong ho toc do|speedometer)\b"), "speedometer"),
    (re.compile(r"\b(dong ho bam gio|stopwatch|timer)\b"), "timer"),
    (re.compile(r"\b(bang diem|scoreboard)\b"), "scoreboard"),
    (re.compile(r"\b(cong to|meter)\b"), "meter"),
)


@dataclass(frozen=True)
class QaQueryPlan:
    answer_source: str = "visual"
    device: str | None = None
    value_type: str | None = None
    temporal_rule: str | None = None
    requires_dense_read: bool = False
    requires_visual_read: bool = False
    requires_multi_frame: bool = False
    prefer_video_consensus: bool = False
    operation: str | None = None
    interval_start_ms: int | None = None
    interval_end_ms: int | None = None
    candidate_values: tuple[str, ...] = ()
    relation_query: str | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def build_qa_query_plan(description: str, question: str) -> QaQueryPlan:
    """Build a conservative plan from explicit words in the request only."""
    folded_question = fold_text(question)
    folded_all = fold_text(f"{description} {question}")
    raw_all = f"{description} {question}".casefold()
    device = "weighing scale" if re.search(r"\bcân\b", raw_all) else next(
        (name for pattern, name in _DEVICES if pattern.search(folded_all)), None,
    )
    asks_numeric = bool(_NUMERIC.search(folded_question))
    asks_visible_text = bool(_VISIBLE_TEXT.search(folded_question))
    chart_reading = bool(_CHART.search(folded_all)) and asks_numeric
    missing_visible = bool(_MISSING_VISIBLE.search(folded_question))
    seconds_match = _FIRST_SECONDS.search(folded_question)
    range_match = _VALUE_RANGE.search(folded_question)
    if missing_visible and seconds_match and range_match:
        seconds_raw = seconds_match.group("vi") or seconds_match.group("en")
        seconds = float(seconds_raw.replace(",", "."))
        range_start = int(range_match.group("start"))
        range_end = int(range_match.group("end"))
        if 0 < seconds <= 3_600 and 0 <= range_start <= range_end and range_end - range_start <= 255:
            return QaQueryPlan(
                answer_source="temporal_set_difference",
                value_type="number",
                temporal_rule="bounded_interval_set_difference",
                requires_visual_read=True,
                requires_multi_frame=True,
                prefer_video_consensus=True,
                operation="set_difference",
                interval_start_ms=0,
                interval_end_ms=round(seconds * 1_000),
                candidate_values=tuple(str(value) for value in range(range_start, range_end + 1)),
                relation_query="numbered objects visible across the requested time interval",
            )
    if chart_reading:
        return QaQueryPlan(
            answer_source="chart_reading",
            value_type="number",
            temporal_rule="best_legible_chart",
            requires_visual_read=True,
            prefer_video_consensus=True,
            operation="read_chart_relation",
            relation_query="close-up of a chart with labeled curves and readable axis values",
        )
    # "bao nhiêu" alone is not enough to infer OCR. It becomes a display
    # reading only when an explicit visual device/display is also present.
    display_value = device is not None and (asks_numeric or asks_visible_text)
    if not display_value:
        visible_letters = bool(re.search(
            r"\b(dong chu|chu neon|bien hieu|logo|text|lettering|neon|sign reads)\b",
            folded_question,
        ))
        visual_count = bool(re.search(
            r"\b(may|bao nhieu|how many|count)\b", folded_question,
        )) and bool(re.search(
            r"\b(màu|màu sắc|color|colours?|objects?|vật|chiếc|quả|con|hình)\b",
            unicodedata.normalize('NFC', question.casefold()),
        ))
        if visible_letters or visual_count:
            return QaQueryPlan(
                answer_source="visible_text" if visible_letters else "visual_count",
                value_type="text" if visible_letters else "number",
                requires_visual_read=True,
                operation="read_visible_text" if visible_letters else "count_visible",
            )
        return QaQueryPlan()
    temporal_rule = "last_before_scene_change" if _LAST.search(folded_question) else "best_visible_reading"
    return QaQueryPlan(
        answer_source="device_display",
        device=device,
        value_type="number" if asks_numeric else "text",
        temporal_rule=temporal_rule,
        requires_dense_read=True,
        relation_query=f"close-up of the display attached to the {device}",
    )


def relation_preserving_queries(description: str, question: str) -> tuple[str, ...]:
    plan = build_qa_query_plan(description, question)
    return (plan.relation_query,) if plan.relation_query else ()


def question_prefers_visual_text(question: str, description: str = "") -> bool:
    return build_qa_query_plan(description, question).answer_source in {
        "device_display", "chart_reading", "temporal_set_difference", "visible_text", "visual_count",
    }
