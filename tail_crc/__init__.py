"""Single-rollout answer-span scoring and family-level CRC calibration."""
from fractions import Fraction
import math
from statistics import mean

__all__ = ["tail10_nll", "crc_threshold"]


def tail10_nll(logp, valid=True):
    """Mean of the largest ceil(L/10) NLLs of the L sampled answer tokens.

    Supply sampled-token natural log probabilities from after the first
    end-of-thinking delimiter to before EOS, including sampled explanation
    tokens. Omit prefill and injected tokens (including delimiters/newlines).
    Natural closure and budget forcing use the same span. An invalid output
    or an empty span receives +inf. Missing or nonfinite log probabilities
    on an otherwise valid span are input errors.
    """
    if not isinstance(valid, bool):
        raise TypeError("valid must be a boolean")
    if not valid:
        return math.inf
    surprises = []
    for value in logp:
        if isinstance(value, bool) or not math.isfinite(value) or value > 0:
            raise ValueError("a sampled-token log probability must be finite and <= 0")
        surprises.append(-float(value))
    if not surprises:
        return math.inf
    k = (len(surprises) + 9) // 10
    return mean(sorted(surprises, reverse=True)[:k])


def _calibration_rank(n, alpha):
    # Interpret decimal targets exactly, including when (n + 1) * alpha is an integer.
    risk = alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))
    if not 0 < risk < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    return (n + 1) * risk.numerator // risk.denominator


def _threshold_at_rank(sorted_wrong, j):
    if j == 0:
        return -math.inf
    return sorted_wrong[j - 1] if j <= len(sorted_wrong) else math.inf


def crc_threshold(cal, alpha):
    """Fit one strict score cutoff for a fixed solver, seed, decoder and score.

    Give one (grade, score) pair per CAL family. Grades are R, W and U. Larger
    scores indicate greater risk; +inf marks an unusable score and is required
    for U. All grades count toward n. Only W scores enter the order statistic.
    Alpha is in (0, 1); decimal strings and Fractions preserve exact boundaries.
    With j = floor((n+1)*alpha), return -inf for j=0 (including an empty CAL),
    +inf for fewer than j wrong answers, or the j-th smallest wrong score.
    Submit a valid test answer exactly when its score is strictly below the cut.
    """
    wrong = []
    n = 0
    for n, (grade, score) in enumerate(cal, 1):
        if grade not in ("R", "W", "U"):
            raise ValueError(f"unknown calibration grade {grade!r}")
        if isinstance(score, bool) or math.isnan(score) or score == -math.inf:
            raise ValueError("a calibration score must be finite or +inf")
        if grade == "U" and score != math.inf:
            raise ValueError("use +inf as the score for grade U")
        if grade == "W":
            wrong.append(score)
    return _threshold_at_rank(sorted(wrong), _calibration_rank(n, alpha))
