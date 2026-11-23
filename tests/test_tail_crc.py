import math
import unittest
from fractions import Fraction

from tail_crc import crc_threshold, tail10_nll


class AnswerScoreTests(unittest.TestCase):
    def test_round_up_tail_length_and_rank_whole_span(self):
        self.assertEqual(tail10_nll([-3]), 3)
        self.assertEqual(tail10_nll([-8] + [-1] * 9), 8)
        self.assertEqual(tail10_nll([-8] + [-1] * 10), 4.5)
        self.assertEqual(tail10_nll([-8, -8] + [-1] * 9), 8)

    def test_sampled_answer_inputs_after_natural_and_forced_close(self):
        # Synthetic decoder traces. The caller knows token provenance and slices
        # the trace before passing sampled answer logprobs into the public API.
        prefix = [("prefill", None), ("sampled", -30)]
        answer = [("sampled", -8)] + [("sampled", -1)] * 10 + [("EOS", -20)]
        for close in [[("CLOSE", -12)], [("CLOSE", None), ("injected-newline", None)]]:
            trace = prefix + close + answer
            first_close = next(i for i, (role, _) in enumerate(trace) if role == "CLOSE")
            eos = next(i for i, (role, _) in enumerate(trace) if role == "EOS")
            span = (logp for role, logp in trace[first_close + 1:eos] if role == "sampled")
            self.assertEqual(tail10_nll(span), 4.5)

    def test_empty_and_invalid_outputs(self):
        self.assertEqual(tail10_nll([]), math.inf)
        self.assertEqual(tail10_nll(None, valid=False), math.inf)

    def test_unusable_sampled_logprobs_fail(self):
        for invalid in (None, True, math.nan, math.inf, -math.inf, 0.001, "-0.5"):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                tail10_nll([-2, invalid])
        with self.assertRaises(TypeError):
            tail10_nll(None)


class CalibrationTests(unittest.TestCase):
    def test_all_families_count_only_wrong_scores_are_ranked(self):
        cal = [("W", 3), ("R", -1000), ("R", 1000), ("U", math.inf), ("W", 1)]
        self.assertEqual(crc_threshold(iter(cal), Fraction(1, 3)), 3)  # m = j
        self.assertEqual(crc_threshold(cal, Fraction(1, 2)), math.inf)

    def test_ties_rejected_at_strict_cut(self):
        cal = [("W", 1), ("W", 1), ("W", 1), ("R", 0)]
        cut = crc_threshold(cal, ".6")
        self.assertEqual(cut, 1)
        self.assertEqual([grade for grade, score in cal if score < cut], ["R"])

    def test_no_calibration_wrong_answers(self):
        cal = [("U", math.inf)] * 4
        self.assertEqual(crc_threshold([], ".9"), -math.inf)
        self.assertEqual(crc_threshold(cal, ".1"), -math.inf)
        cut = crc_threshold(cal, ".2")
        self.assertEqual(cut, math.inf)
        self.assertFalse(any(score < cut for _, score in cal))

    def test_decimal_boundaries(self):
        cal = [("W", s) for s in range(99)]
        self.assertEqual(crc_threshold(cal, 0.58), 57)
        self.assertEqual(crc_threshold(cal, "0.579999999999999999999999"), 56)
        self.assertEqual(crc_threshold(cal, math.nextafter(.58, 0)), 56)
        self.assertEqual(crc_threshold(cal, ".00999999999999999999999"), -math.inf)

    def test_paper_rank_uses_300_families(self):
        cal = [("W", s) for s in range(300)]
        cut = crc_threshold(cal, ".065")
        self.assertEqual(cut, 18)
        self.assertEqual(sum(s < cut for _, s in cal), 18)

    def test_per_seed_calibration_changes_the_decision(self):
        seed1 = [("R", .1), ("W", .2), ("W", .5), ("U", math.inf)]
        seed2 = [("R", .1), ("W", .6), ("W", .9), ("U", math.inf)]
        cuts = [crc_threshold(seed, ".4") for seed in (seed1, seed2)]
        self.assertEqual(cuts, [.5, .9])
        self.assertEqual([.7 < cut for cut in cuts], [False, True])
        self.assertEqual(crc_threshold(seed1 + seed2, ".4"), .6)

    def test_invalid_labels_scores_and_targets(self):
        for row in (("?", 0), ("R", None), ("R", math.nan), ("R", "0"),
                    ("U", .1), ("W", True), ("W", -math.inf)):
            with self.subTest(row=row), self.assertRaises((TypeError, ValueError)):
                crc_threshold([row], ".5")
        for alpha in (0, 1, "1.0001", -2, None, True, math.nan, math.inf):
            with self.subTest(alpha=alpha), self.assertRaises((TypeError, ValueError)):
                crc_threshold([], alpha)


if __name__ == "__main__":
    unittest.main()
