import math
import unittest

import numpy as np

from replay.figure_data import resplit_thresholds
from replay.main_panels import Panel
from replay.uncertainty import crc_pass
from tail_crc import crc_threshold


class ReplayAdapterTests(unittest.TestCase):
    def test_scalar_and_batched_cuts_match_for_all_ranks(self):
        cals = [
            [("W", .2), ("W", .2), ("R", .05), ("U", math.inf)],
            [("U", math.inf)] * 4,
            [("W", .8), ("W", .7), ("W", .4), ("R", .75)],
        ]
        wrong = np.array([[s if g == "W" else math.inf for g, s in cal] for cal in cals])
        candidates = np.array([.1, .2, .7, .8, .9, math.inf])
        for alpha in (".1", ".2", ".3999999999999", ".4", ".6", ".8"):
            expected = [crc_threshold(cal, alpha) for cal in cals]
            np.testing.assert_array_equal(resplit_thresholds(wrong, alpha), expected)
            for cal, cut in zip(cals, expected):
                records = [dict(family=i, vs=g, x=s) for i, (g, s) in enumerate(cal)]
                self.assertEqual(Panel(records).cut("x", alpha), cut)
                np.testing.assert_array_equal(crc_pass(records, candidates, "x", alpha),
                                              candidates < cut)
