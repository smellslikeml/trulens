"""Unit tests for continuous verification scoring (LLM-as-a-Verifier).

Covers the parameter-free estimators in
[verifier_score][trulens.feedback.verifier_score] and their wiring into
[generate_verifier_score][trulens.feedback.llm_provider.LLMProvider.generate_verifier_score].
"""

from __future__ import annotations

import math
from typing import List, Optional
from unittest import TestCase

from trulens.feedback import llm_provider
from trulens.feedback import verifier_score


class VerifierScoreEstimatorTests(TestCase):
    """Tests for the standalone expectation estimators."""

    def test_expected_score_uses_full_distribution(self):
        """Expectation blends the score tokens, not just the argmax."""

        # Two tokens with equal logprob (log 0.5 each) -> expectation 7.5,
        # which normalizes to 0.75 on a 0-10 scale. A discrete argmax parse
        # would instead collapse to a single integer.
        logprobs = {" 7": math.log(0.5), " 8": math.log(0.5)}
        score = verifier_score.expected_score_from_logprobs(
            logprobs, min_score_val=0, max_score_val=10
        )
        self.assertAlmostEqual(score, 0.75, places=6)

    def test_expected_score_ignores_out_of_range_tokens(self):
        """Non-integer and out-of-range tokens are dropped before softmax."""

        logprobs = [("7", math.log(0.9)), ("cat", math.log(0.05)), ("99", -1.0)]
        score = verifier_score.expected_score_from_logprobs(
            logprobs, min_score_val=0, max_score_val=10
        )
        # Only "7" survives, so the expectation is exactly 7 -> 0.7.
        self.assertAlmostEqual(score, 0.7, places=6)

    def test_expected_score_invalid_when_no_valid_token(self):
        """Returns the sentinel when no scoring token can be parsed."""

        score = verifier_score.expected_score_from_logprobs(
            {"foo": -0.1, "bar": -0.2}, min_score_val=0, max_score_val=10
        )
        self.assertEqual(score, verifier_score.INVALID_SCORE)

    def test_aggregate_repeated_scores_averages_valid(self):
        """Repeated-evaluation aggregation is the mean of valid samples."""

        result = verifier_score.aggregate_repeated_scores([0.2, 0.4, 0.6])
        self.assertAlmostEqual(result, 0.4, places=6)

    def test_aggregate_repeated_scores_drops_sentinels(self):
        """Invalid (negative) samples are excluded from the mean."""

        result = verifier_score.aggregate_repeated_scores([
            0.5,
            verifier_score.INVALID_SCORE,
            0.7,
        ])
        self.assertAlmostEqual(result, 0.6, places=6)

    def test_aggregate_repeated_scores_all_invalid(self):
        """Returns the sentinel when nothing is valid."""

        result = verifier_score.aggregate_repeated_scores([
            verifier_score.INVALID_SCORE
        ])
        self.assertEqual(result, verifier_score.INVALID_SCORE)


class _ScriptedProvider(llm_provider.LLMProvider):
    """LLMProvider whose ``generate_score`` replays a scripted sequence.

    Exercises the ``generate_verifier_score`` wiring on the real base class
    without needing a networked endpoint.
    """

    model_config = {"extra": "allow"}

    scripted_scores: Optional[List[float]] = None

    def __init__(self, scripted_scores: List[float], **kwargs):
        super().__init__(endpoint=None, model_engine="scripted", **kwargs)
        self.scripted_scores = list(scripted_scores)
        self._cursor = 0

    def generate_score(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 10,
        temperature: float = 0.0,
    ) -> float:
        score = self.scripted_scores[self._cursor % len(self.scripted_scores)]
        self._cursor += 1
        return score


class GenerateVerifierScoreWiringTests(TestCase):
    """Tests for the LLMProvider.generate_verifier_score call site."""

    def test_repeated_evaluation_averages_discrete_scores(self):
        """n_samples repeated evaluations are averaged into one score."""

        provider = _ScriptedProvider(scripted_scores=[0.2, 0.4, 0.9])
        result = provider.generate_verifier_score(
            system_prompt="rate this", n_samples=3
        )
        self.assertAlmostEqual(result, (0.2 + 0.4 + 0.9) / 3, places=6)

    def test_single_sample_matches_discrete_score(self):
        """With n_samples=1 the verifier score equals the discrete score."""

        provider = _ScriptedProvider(scripted_scores=[0.3])
        result = provider.generate_verifier_score(system_prompt="rate this")
        self.assertAlmostEqual(result, 0.3, places=6)

    def test_tuple_return_from_generate_score_is_normalized(self):
        """A (score, reasons) return from generate_score is handled."""

        class _TupleProvider(_ScriptedProvider):
            def generate_score(self, *args, **kwargs):
                return (0.42, {"reason": "because"})

        provider = _TupleProvider(scripted_scores=[0.0])
        result = provider.generate_verifier_score(
            system_prompt="rate this", n_samples=2
        )
        self.assertAlmostEqual(result, 0.42, places=6)

    def test_logprobs_path_uses_exact_expectation(self):
        """When score_logprobs is supplied, the exact estimator is used."""

        # generate_score would raise (endpoint is None); supplying logprobs
        # must bypass it entirely and compute the exact expectation.
        provider = _ScriptedProvider(scripted_scores=[])
        result = provider.generate_verifier_score(
            system_prompt="rate this",
            min_score_val=0,
            max_score_val=10,
            score_logprobs={" 6": math.log(0.5), " 8": math.log(0.5)},
        )
        self.assertAlmostEqual(result, 0.7, places=6)
