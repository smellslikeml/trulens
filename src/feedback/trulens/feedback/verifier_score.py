"""Continuous verification scoring for LLM-based judges.

Utilities that turn an LLM judge's *distribution* over scoring tokens into a
single fine-grained, calibrated score in ``[0, 1]``.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). A standard LLM judge emits one discrete rating; this
module instead estimates the *expectation* of the rating under the judge's
output distribution. That probabilistic formulation separates positive and
negative solutions more cleanly and yields more calibrated comparisons than a
single discrete parse.

Two parameter-free estimators of that expectation are provided:

* [expected_score_from_logprobs][trulens.feedback.verifier_score.expected_score_from_logprobs]
  computes the expectation exactly from the top-logprobs of the scoring token,
  when the provider exposes them (e.g. OpenAI).
* [aggregate_repeated_scores][trulens.feedback.verifier_score.aggregate_repeated_scores]
  estimates the same expectation by Monte-Carlo averaging over repeated
  discrete evaluations, for providers that do not expose logprobs.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Sentinel returned across the feedback stack when a score cannot be parsed.
INVALID_SCORE: float = -1.0

# A distribution over candidate scoring tokens: either a mapping of token to
# log-probability or a sequence of (token, log-probability) pairs.
LogprobDistribution = Union[
    Mapping[str, float],
    Sequence[Tuple[str, float]],
]


def _iter_logprob_items(
    score_logprobs: LogprobDistribution,
) -> Iterator[Tuple[str, float]]:
    """Yield ``(token, logprob)`` pairs from a mapping or a sequence."""

    if isinstance(score_logprobs, Mapping):
        yield from score_logprobs.items()
    else:
        yield from score_logprobs


def _parse_score_token(token: str) -> Optional[int]:
    """Parse an integer score from a raw scoring token.

    Provider tokens frequently carry surrounding whitespace (e.g. ``" 7"``),
    so the token is stripped before parsing. Returns ``None`` if the token is
    not an integer.
    """

    try:
        return int(token.strip())
    except (TypeError, ValueError):
        return None


def _softmax(logprobs: Sequence[float]) -> list[float]:
    """Numerically stable softmax over log-probabilities."""

    shift = max(logprobs)
    exps = [math.exp(lp - shift) for lp in logprobs]
    total = sum(exps)
    return [e / total for e in exps]


def expected_score_from_logprobs(
    score_logprobs: LogprobDistribution,
    min_score_val: int = 0,
    max_score_val: int = 10,
) -> float:
    """Continuous score as the expectation over scoring-token logits.

    This is the core LLM-as-a-Verifier mechanism: rather than taking the single
    most-likely rating, it computes ``E[score] = sum_k k * P(k)`` where ``P``
    is the judge's (renormalized) distribution over the integer scoring tokens,
    then normalizes onto a ``[0, 1]`` scale.

    Only tokens that parse to an integer inside ``[min_score_val,
    max_score_val]`` contribute; their log-probabilities are renormalized with
    a softmax so they form a proper distribution over the valid scores.

    Args:
        score_logprobs: The judge's top-logprobs for the scoring token, as a
            mapping of ``token -> logprob`` or a sequence of
            ``(token, logprob)`` pairs.
        min_score_val: The minimum score value on the judge's scale.
        max_score_val: The maximum score value on the judge's scale.

    Returns:
        The expected score normalized to ``[0, 1]``, or
        [INVALID_SCORE][trulens.feedback.verifier_score.INVALID_SCORE] if no
        valid scoring token is present.
    """

    if max_score_val <= min_score_val:
        raise ValueError("Max score must be greater than min score.")

    values: list[int] = []
    logprobs: list[float] = []
    for token, logprob in _iter_logprob_items(score_logprobs):
        value = _parse_score_token(token)
        if value is None or not (min_score_val <= value <= max_score_val):
            continue
        values.append(value)
        logprobs.append(float(logprob))

    if not values:
        logger.warning(
            "No valid scoring token in [%s, %s] found in logprobs.",
            min_score_val,
            max_score_val,
        )
        return INVALID_SCORE

    probs = _softmax(logprobs)
    expectation = sum(p * v for p, v in zip(probs, values))
    return (expectation - min_score_val) / (max_score_val - min_score_val)


def aggregate_repeated_scores(
    scores: Sequence[float],
) -> float:
    """Monte-Carlo estimate of the expected score from repeated evaluations.

    For providers that do not expose token logprobs, the verifier's expectation
    over the scoring distribution is estimated by averaging several independent
    (already normalized to ``[0, 1]``) discrete evaluations. This realizes the
    paper's "repeated evaluation" scaling axis, reducing variance to produce a
    finer-grained, better-calibrated score than any single discrete parse.

    Invalid samples (negative sentinels such as
    [INVALID_SCORE][trulens.feedback.verifier_score.INVALID_SCORE]) are dropped
    before averaging.

    Args:
        scores: Normalized scores in ``[0, 1]`` from repeated evaluations.

    Returns:
        The mean of the valid samples, or
        [INVALID_SCORE][trulens.feedback.verifier_score.INVALID_SCORE] if none
        are valid.
    """

    valid = [s for s in scores if s is not None and s >= 0.0]
    if not valid:
        return INVALID_SCORE
    return sum(valid) / len(valid)
