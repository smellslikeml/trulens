"""Hierarchical reasoning outline for OTEL trace spans.

Adapted from "ReasoningLens: Hierarchical Visualization and Diagnostic
Auditing for Large Reasoning Models" (arXiv:2606.23404).

Long Chain-of-Thought and agentic traces captured by TruLens tend to bury the
high-level *strategy* of a reasoning model under a wall of low-level
*execution* spans (LLM generations, tool calls, retrievals). This module
restructures a flat list of OTEL spans into an interactive hierarchy that
separates strategy from execution, and synthesizes a compact *reasoning
profile* (span-type histogram, depth/breadth, error count) so a record's
reasoning shape can be audited at a glance before drilling into the full
timeline.

This is the deterministic, structural slice of ReasoningLens. The paper's
LLM-driven "agentic auditor" is intentionally out of scope here -- the
structuring and profiling layers deliver the interpretability win without
adding model calls to the dashboard render path. Structural error signals are
surfaced from span status instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any

from trulens.otel.semconv.trace import SpanAttributes

if TYPE_CHECKING:
    from trulens.dashboard.components.record_viewer_otel import OtelSpan

_SpanType = SpanAttributes.SpanType

# Levels of the reasoning hierarchy. "strategy" spans orchestrate the work;
# "execution" spans are the concrete low-level calls that carry it out.
STRATEGY = "strategy"
EXECUTION = "execution"
EVAL = "eval"
OTHER = "other"

# High-level orchestration / planning span types.
STRATEGY_SPAN_TYPES = frozenset({
    _SpanType.RECORD_ROOT.value,
    _SpanType.AGENT.value,
    _SpanType.GRAPH_TASK.value,
    _SpanType.GRAPH_NODE.value,
    _SpanType.WORKFLOW_STEP.value,
})

# Low-level execution span types -- the procedural text the paper warns about.
EXECUTION_SPAN_TYPES = frozenset({
    _SpanType.GENERATION.value,
    _SpanType.RETRIEVAL.value,
    _SpanType.TOOL.value,
    _SpanType.RERANKER.value,
    _SpanType.MCP.value,
    _SpanType.GUARDRAIL.value,
})

# Feedback-evaluation span types.
EVAL_SPAN_TYPES = frozenset({
    _SpanType.EVAL_ROOT.value,
    _SpanType.EVAL.value,
})

# Glyphs used when rendering an outline as indented text.
LEVEL_MARKERS: dict[str, str] = {
    STRATEGY: "▸",
    EXECUTION: "•",
    EVAL: "✓",
    OTHER: "·",
}


def classify_span_level(span_type: str) -> str:
    """Map a span type to its reasoning-hierarchy level.

    Args:
        span_type: The ``ai.observability.span_type`` value of a span.

    Returns:
        One of ``STRATEGY``, ``EXECUTION``, ``EVAL`` or ``OTHER``.
    """
    if span_type in STRATEGY_SPAN_TYPES:
        return STRATEGY
    if span_type in EXECUTION_SPAN_TYPES:
        return EXECUTION
    if span_type in EVAL_SPAN_TYPES:
        return EVAL
    return OTHER


def _span_type_of(span: Mapping[str, Any]) -> str:
    attributes = span.get("record_attributes") or {}
    return str(
        attributes.get(SpanAttributes.SPAN_TYPE, _SpanType.UNKNOWN.value)
    )


def is_error_status(status: str) -> bool:
    """Return True if an OTEL span status string indicates an error."""
    return "error" in str(status).lower()


@dataclass
class OutlineNode:
    """A node in the hierarchical reasoning outline."""

    span_id: str
    name: str
    span_type: str
    level: str
    status: str
    start_timestamp: Any = 0
    depth: int = 0
    children: list[OutlineNode] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return is_error_status(self.status)


def build_reasoning_outline(
    spans: list[OtelSpan],
) -> list[OutlineNode]:
    """Build a parent/child outline tree from a flat list of OTEL spans.

    Spans are linked via ``trace.span_id`` / ``trace.parent_id``. Spans whose
    parent is missing from the list (or absent) become roots, so the outline
    is always complete even for partial traces. Children are ordered by start
    time. Self/cyclic parent references are dropped defensively.

    Args:
        spans: OTEL spans as produced by the dashboard's event-conversion path.

    Returns:
        The root ``OutlineNode`` objects, each populated with ``children`` and
        a computed ``depth``.
    """
    nodes: dict[str, OutlineNode] = {}
    parents: dict[str, str] = {}

    for span in spans:
        trace = span.get("trace") or {}
        span_id = str(trace.get("span_id", ""))
        if not span_id:
            continue
        record = span.get("record") or {}
        span_type = _span_type_of(span)
        nodes[span_id] = OutlineNode(
            span_id=span_id,
            name=str(record.get("name", "")) or "(unnamed span)",
            span_type=span_type,
            level=classify_span_level(span_type),
            status=str(record.get("status", "")),
            start_timestamp=span.get("start_timestamp", 0),
        )
        parents[span_id] = str(trace.get("parent_id", ""))

    roots: list[OutlineNode] = []
    for span_id, node in nodes.items():
        parent_id = parents.get(span_id, "")
        if parent_id and parent_id != span_id and parent_id in nodes:
            nodes[parent_id].children.append(node)
        else:
            roots.append(node)

    def _sort_and_depth(node: OutlineNode, depth: int, seen: set) -> None:
        if node.span_id in seen:
            return
        seen.add(node.span_id)
        node.depth = depth
        node.children.sort(key=lambda c: (c.start_timestamp, c.name))
        for child in node.children:
            _sort_and_depth(child, depth + 1, seen)

    seen: set = set()
    roots.sort(key=lambda n: (n.start_timestamp, n.name))
    for root in roots:
        _sort_and_depth(root, 0, seen)

    return roots


def iter_outline(
    nodes: list[OutlineNode],
) -> Iterator[tuple[OutlineNode, int]]:
    """Pre-order traversal yielding ``(node, depth)`` for rendering."""
    for node in nodes:
        yield node, node.depth
        yield from iter_outline(node.children)


def summarize_reasoning_profile(spans: list[OtelSpan]) -> dict[str, Any]:
    """Synthesize a compact reasoning profile for a trace.

    Args:
        spans: OTEL spans for a single record.

    Returns:
        A dict with overall counts (``total_spans``), per-level and per-type
        histograms (``by_level``, ``by_type``), hierarchy depth (``max_depth``),
        convenience ``strategy_count`` / ``execution_count`` fields, and
        structural error signals (``error_count``, ``has_errors``).
    """
    by_level: dict[str, int] = {STRATEGY: 0, EXECUTION: 0, EVAL: 0, OTHER: 0}
    by_type: dict[str, int] = {}
    error_count = 0

    for span in spans:
        span_type = _span_type_of(span)
        by_level[classify_span_level(span_type)] += 1
        by_type[span_type] = by_type.get(span_type, 0) + 1
        record = span.get("record") or {}
        if is_error_status(str(record.get("status", ""))):
            error_count += 1

    roots = build_reasoning_outline(spans)
    max_depth = 0
    for _node, depth in iter_outline(roots):
        max_depth = max(max_depth, depth)

    return {
        "total_spans": len(spans),
        "by_level": by_level,
        "by_type": by_type,
        "max_depth": max_depth,
        "strategy_count": by_level[STRATEGY],
        "execution_count": by_level[EXECUTION],
        "error_count": error_count,
        "has_errors": error_count > 0,
    }
