"""Unit tests for the hierarchical reasoning outline.

These tests exercise the ReasoningLens-inspired structuring/profiling layer
(``trulens.dashboard.utils.reasoning_outline``) end-to-end against the
dashboard's *existing* OTEL span-conversion path
(``_convert_events_to_otel_spans``), so they prove the new code consumes
exactly the span shape the Records dashboard tab produces.
"""

import unittest

import pandas as pd
from trulens.dashboard.utils import reasoning_outline

# Existing (non-new) dashboard module that produces the OtelSpan dicts the
# Records tab feeds into the reasoning outline.
from trulens.dashboard.utils.dashboard_utils import (
    _convert_events_to_otel_spans,
)
from trulens.otel.semconv.trace import SpanAttributes

_SpanType = SpanAttributes.SpanType


def _event_row(
    span_id: str,
    parent_id: str,
    name: str,
    span_type: str,
    start: int,
    status: str = "STATUS_CODE_OK",
):
    """Build one Event ORM-shaped row, as stored in the events table."""
    return {
        "event_id": f"event-{span_id}",
        "record": {
            "name": name,
            "parent_span_id": parent_id,
            "status": status,
        },
        "trace": {
            "trace_id": "trace-0",
            "parent_id": parent_id,
            "span_id": span_id,
        },
        "record_attributes": {SpanAttributes.SPAN_TYPE: span_type},
        "start_timestamp": start,
        "timestamp": start + 1,
    }


def _agentic_trace_spans():
    """An agent trace: record_root -> agent -> {generation, tool*, retrieval}."""
    events_df = pd.DataFrame([
        _event_row("s0", "", "record_root", _SpanType.RECORD_ROOT.value, 1),
        _event_row("s1", "s0", "planner", _SpanType.AGENT.value, 2),
        _event_row("s2", "s1", "llm_call", _SpanType.GENERATION.value, 3),
        _event_row(
            "s3",
            "s1",
            "search_tool",
            _SpanType.TOOL.value,
            4,
            status="STATUS_CODE_ERROR",
        ),
        _event_row("s4", "s1", "vector_lookup", _SpanType.RETRIEVAL.value, 5),
    ])
    # Use the existing dashboard conversion path, not a hand-rolled dict.
    return _convert_events_to_otel_spans(events_df)


class TestSpanClassification(unittest.TestCase):
    def test_strategy_vs_execution_levels(self):
        self.assertEqual(
            reasoning_outline.classify_span_level(_SpanType.AGENT.value),
            reasoning_outline.STRATEGY,
        )
        self.assertEqual(
            reasoning_outline.classify_span_level(_SpanType.GENERATION.value),
            reasoning_outline.EXECUTION,
        )
        self.assertEqual(
            reasoning_outline.classify_span_level(_SpanType.EVAL.value),
            reasoning_outline.EVAL,
        )
        self.assertEqual(
            reasoning_outline.classify_span_level("something_unknown"),
            reasoning_outline.OTHER,
        )

    def test_error_status_detection(self):
        self.assertTrue(reasoning_outline.is_error_status("STATUS_CODE_ERROR"))
        self.assertFalse(reasoning_outline.is_error_status("STATUS_CODE_OK"))


class TestBuildReasoningOutline(unittest.TestCase):
    def test_hierarchy_from_converted_spans(self):
        spans = _agentic_trace_spans()
        self.assertEqual(len(spans), 5)

        roots = reasoning_outline.build_reasoning_outline(spans)

        # Single record_root at the top of the hierarchy.
        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertEqual(root.span_type, _SpanType.RECORD_ROOT.value)
        self.assertEqual(root.level, reasoning_outline.STRATEGY)
        self.assertEqual(root.depth, 0)

        # The agent is the only child of the root.
        self.assertEqual(len(root.children), 1)
        agent = root.children[0]
        self.assertEqual(agent.span_type, _SpanType.AGENT.value)
        self.assertEqual(agent.depth, 1)

        # The three execution spans hang off the agent, ordered by start time.
        child_types = [c.span_type for c in agent.children]
        self.assertEqual(
            child_types,
            [
                _SpanType.GENERATION.value,
                _SpanType.TOOL.value,
                _SpanType.RETRIEVAL.value,
            ],
        )
        for child in agent.children:
            self.assertEqual(child.depth, 2)

    def test_orphan_spans_become_roots(self):
        # Parent id points at a span that is not present -> treated as a root.
        events_df = pd.DataFrame([
            _event_row("a", "missing", "orphan", _SpanType.TOOL.value, 1),
        ])
        spans = _convert_events_to_otel_spans(events_df)
        roots = reasoning_outline.build_reasoning_outline(spans)
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].span_id, "a")

    def test_iter_outline_preorder(self):
        spans = _agentic_trace_spans()
        roots = reasoning_outline.build_reasoning_outline(spans)
        ordered = [
            node.name for node, _ in reasoning_outline.iter_outline(roots)
        ]
        self.assertEqual(
            ordered,
            [
                "record_root",
                "planner",
                "llm_call",
                "search_tool",
                "vector_lookup",
            ],
        )


class TestReasoningProfile(unittest.TestCase):
    def test_profile_counts_and_errors(self):
        spans = _agentic_trace_spans()
        profile = reasoning_outline.summarize_reasoning_profile(spans)

        self.assertEqual(profile["total_spans"], 5)
        # record_root + agent.
        self.assertEqual(profile["strategy_count"], 2)
        # generation + tool + retrieval.
        self.assertEqual(profile["execution_count"], 3)
        self.assertEqual(profile["max_depth"], 2)
        self.assertEqual(profile["error_count"], 1)
        self.assertTrue(profile["has_errors"])
        self.assertEqual(profile["by_type"][_SpanType.GENERATION.value], 1)

    def test_empty_trace_profile(self):
        profile = reasoning_outline.summarize_reasoning_profile([])
        self.assertEqual(profile["total_spans"], 0)
        self.assertEqual(profile["max_depth"], 0)
        self.assertFalse(profile["has_errors"])


if __name__ == "__main__":
    unittest.main()
