"""Check Theorem 1 and the exact LTT test on finite examples.

Run from the repository root with python -m theory.verify_crc_theory.
"""
from fractions import Fraction
from itertools import product
import math
from scipy.stats import binom

from tail_crc import crc_threshold


def check_leave_one_out():
    alphabet = [("W", 0.), ("W", 1.), ("W", 2.),
                ("R", math.inf), ("U", math.inf)]
    positive_rank = zero_rank = 0
    for N in range(2, 7):
        alphas = [Fraction(1, 2*N)] + [Fraction(j, N) for j in range(1, N)]
        for data in product(alphabet, repeat=N):
            failures = [0] * N
            for i, (grade, score) in enumerate(data):
                cal = data[:i] + data[i+1:]
                z = sorted(s if y == "W" else math.inf for y, s in cal)
                for j, alpha in enumerate(alphas):
                    cut = crc_threshold(cal, alpha)
                    assert cut == (-math.inf if j == 0 else z[j - 1])
                    failures[j] += grade == "W" and score < cut
            for j, count in enumerate(failures):
                assert count <= j, (data, j, count)
            positive_rank += N - 1
            zero_rank += 1
    return positive_rank, zero_rank


def check_binomial_boundary():
    for n in range(1, 61):
        for alpha in [.1, .15, .2, .3]:
            rejection = [k for k in range(n + 1) if binom.cdf(k, n, alpha) <= .01]
            for true_r in [alpha, min(1., alpha + .01), .5, .8, 1.]:
                if true_r >= alpha:
                    prob = sum(binom.pmf(k, n, true_r) for k in rejection)
                    assert prob <= .01 + 1e-12, (n, alpha, true_r, prob)


if __name__ == "__main__":
    positive, zero = check_leave_one_out()
    print(f"PASS: {positive:,} positive-rank and {zero:,} zero-rank leave-one-out configurations")
    check_binomial_boundary()
    print("PASS: binomial false-issuance probability at and above the null boundary")
    print("Minimum zero-error accepted volumes:",
          [math.ceil(math.log(.01)/math.log(1-a)) for a in [.1, .15, .2, .3]])
