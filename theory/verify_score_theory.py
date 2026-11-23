#!/usr/bin/env python3
"""Reproduce Tail-CRC score-theory numerical checks (Python standard library).

Run from the repository root with python -m theory.verify_score_theory.
No network, external data, third-party packages, or output files are required.
The checks support the analytic proofs; numerical checks do not prove them.
"""

import argparse
import itertools
import json
import math
import random
from collections import Counter
from fractions import Fraction
from statistics import fmean, pvariance

from tail_crc import tail10_nll


TOL = 1e-10


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(a, b, message):
    require(math.isclose(a, b, rel_tol=TOL, abs_tol=TOL),
            f"{message}: {a!r} != {b!r}")


def tail(values, k):
    return fmean(sorted(values, reverse=True)[:k])


def cvar(values, beta):
    """Exact tail mass/rank arithmetic; only score arithmetic uses floats."""
    values = sorted(values, reverse=True)
    mass = beta * len(values)
    whole = mass.numerator // mass.denominator
    fraction = float(mass - whole)
    boundary = fraction * values[whole] if whole < len(values) else 0.0
    return (sum(values[:whole]) + boundary) / float(mass)


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p)


def surprise_variance(probabilities):
    h = entropy(probabilities)
    return sum(p * (-math.log(p) - h) ** 2 for p in probabilities if p)


def psi(value):
    return float(value > 0) + 0.5 * float(value == 0)


def check_tail_geometry(seed, max_length):
    rng = random.Random(seed)
    counts = Counter()
    for length in range(1, max_length + 1):
        for k in range(1, length + 1):
            x = [rng.gauss(0, 1) for _ in range(length)]
            z = [rng.gauss(0, 1) for _ in range(length)]
            mean_z = fmean(z)
            lhs = abs(tail([a + b for a, b in zip(x, z)], k)
                      - tail(x, k) - mean_z)
            rhs = math.sqrt(max(0.0, 1 / k - 1 / length)
                            * sum((v - mean_z) ** 2 for v in z))
            require(lhs <= rhs + TOL, f"Pathwise bound: L={length}, k={k}")
            sharp = [float(i < k) / k - 1 / length for i in range(length)]
            near(tail(sharp, k), 1 / k - 1 / length, "Sharp constant")
            counts["pathwise"] += 1
            counts["sharpness"] += 1
        # Fraction avoids spurious ceil errors such as ceil(.2*3*5)=4.
        for beta in map(Fraction, [".1", ".2", ".33", ".7", "1"]):
            x = [rng.gauss(0, 1) for _ in range(length)]
            mass = beta * length
            k = math.ceil(mass)
            remainder = float(k - mass)
            empirical = tail(x, k)
            if beta == Fraction(1, 10):
                # Translate the finite Gaussian fixture to nonnegative NLLs.
                low = min(x)
                near(tail10_nll(low - v for v in x), empirical - low,
                     "Public score and top-k geometry")
            exact = cvar(x, beta)
            kth = sorted(x, reverse=True)[k - 1]
            near(exact - empirical, remainder / k * (exact - kth),
                 "Fractional CVaR rounding identity")
            counts["rounding"] += 1
            for copies in [1, 2, 3, 10]:
                replicated = [v for v in x for _ in range(copies)]
                difference = tail(replicated, math.ceil(beta * copies * length)) - empirical
                bound = remainder / k * (max(x) - min(x))
                require(-TOL <= difference <= bound + TOL,
                        f"Replication: L={length}, beta={beta}, c={copies}")
                counts["replication"] += 1
    x = [1.0] + [0.0] * 10
    near(tail(x, 2), 0.5, "Short-span score")
    near(tail([v for v in x for _ in range(10)], 11), 10 / 11,
         "Failure of exact rounded-score replication invariance")
    return dict(counts)


def check_length_counterexamples():
    right = [(0.8, 2, 1), (0.2, 4, 4)]
    wrong = [(0.8, 1, 3), (0.2, 3, 0.8)]
    def auc(score):
        return sum(pr * pw * psi(score(mw, uw) - score(mr, ur))
                   for pr, mr, ur in right for pw, mw, uw in wrong)
    # Right lengths stochastically dominate wrong lengths at every support cut.
    for cut in [1, 2, 3, 4]:
        require(sum(p for p, m, _ in right if m <= cut)
                <= sum(p for p, m, _ in wrong if m <= cut) + TOL, "Length FOSD")
    near(auc(lambda m, u: u), 0.64, "FOSD counterexample mean AUROC")
    near(auc(lambda m, u: m * u), 0.8, "FOSD counterexample sum AUROC")
    median_counter = sum(pd * pb * psi(d - b)
                         for pd, d in [(0.51, 0.1), (0.49, 1.0)]
                         for pb, b in [(0.49, 0.0), (0.51, math.log(2))])
    near(median_counter, 0.7399, "Median-gap counterexample")
    near(0.51 * 1 + 0.49 * 0.1, 0.559, "Bounded/sub-Gaussian mean gap")
    require(0.559 < math.log(2), "Sub-Gaussian reversal premise")
    return {"fosd_mean_auc": 0.64, "fosd_sum_auc": 0.8,
            "median_gap_sum_auc": median_counter}


def check_bounded_model():
    means = [3.0, 3.0, 1.0, 1.0]
    amplitudes = [0.2] * 4
    signs = list(itertools.product([-1, 1], repeat=4))
    pi_right, pi_wrong = 0.6, 0.3
    checked = 0
    for shift in [0.1, 0.3, 1.0]:
        delta = [shift, shift, 0.0, 0.0]
        total_signal = sum(delta)
        for indices in [[0, 1], [0, 1, 2, 3]]:
            right, wrong = [], []
            for sign in signs:
                x = [m + a * s for m, a, s in zip(means, amplitudes, sign)]
                y = [v + d for v, d in zip(x, delta)]
                right.append(fmean(x[i] for i in indices))
                wrong.append(fmean(y[i] for i in indices))
                if len(indices) == 2:
                    near(right[-1], tail(x, 2), "Stable critical selection, R")
                    near(wrong[-1], tail(y, 2), "Stable critical selection, W")
            amplitude_sum = sum(amplitudes[i] ** 2 for i in indices)
            separation = (fmean(wrong) - fmean(right)) / math.sqrt(pvariance(wrong) + pvariance(right))
            near(separation, total_signal / math.sqrt(2 * amplitude_sum), "Exact SNR")
            for alpha in [0.001, 0.01, 0.05, 0.1, 0.2]:
                radius = math.sqrt(2 * amplitude_sum * math.log(pi_wrong / alpha))
                threshold = fmean(wrong) - radius / len(indices)
                retained_wrong = pi_wrong * fmean(float(s < threshold) for s in wrong)
                retained_right = pi_right * fmean(float(s < threshold) for s in right)
                margin = max(0.0, total_signal / math.sqrt(amplitude_sum)
                             - math.sqrt(2 * math.log(pi_wrong / alpha)))
                lower_bound = pi_right * (1 - math.exp(-margin * margin / 2))
                require(retained_wrong <= alpha + TOL, "Wrong-budget threshold")
                require(retained_right >= lower_bound - TOL, "Correct-retention bound")
                checked += 1
    return {"enumerated_operating_points": checked, "noise_states_per_class": len(signs)}


def check_surprise_decomposition():
    p = [0.6, 0.2, 0.1, 0.1]
    retained_mass = sum(p[:2])
    q = [v / retained_mass for v in p[:2]]
    c = sum(qi * -math.log(pi) for qi, pi in zip(q, p))
    v = sum(qi * (-math.log(pi) - c) ** 2 for qi, pi in zip(q, p))
    near(c, entropy(q) - math.log(retained_mass), "Truncated-law cross entropy")
    near(v, surprise_variance(q), "Truncated-law variance")
    require(c < entropy(p), "Nonzero negative decoder drift")
    # Genuine autoregression with future distributions depending on past draws.
    states = [(1.0, (), 0.0, 0.0)]
    for step in range(3):
        extended = []
        for prob, history, residual, variance_sum in states:
            first = 0.7 if step == 0 else (0.8 if sum(history) % 2 == 0 else 0.4)
            dist = [first, 1 - first]
            h, var = entropy(dist), surprise_variance(dist)
            for token, chance in enumerate(dist):
                extended.append((prob * chance, history + (token,),
                                 residual - math.log(chance) - h, variance_sum + var))
        states = extended
    near(sum(p * r for p, _, r, _ in states), 0, "Adaptive MDS mean")
    near(sum(p * r * r for p, _, r, _ in states),
         sum(p * v for p, _, _, v in states), "Adaptive MDS isometry")
    a = math.log(2) / 3
    stopped = [(1 / 3, 2 * a), (2 / 9, a / 2), (4 / 9, -a)]
    near(sum(p * v for p, v in stopped), a / 3, "Random-normalization bias")
    near(sum(p * v * v for p, v in stopped), 11 * a * a / 6, "Stopped second moment")
    require(not math.isclose(11 * a * a / 6, 4 * a * a / 3), "Naive stopped isometry fails")
    return {"decoder_drift": c - entropy(p), "adaptive_paths": len(states),
            "random_normalization_bias": a / 3}


def check_jensen_counterexamples():
    p1, p2 = [2 / 3, 1 / 3], [0.3, 0.3, 0.2, 0.2]
    expected_max = sum(a * b * max(-math.log(a), -math.log(b)) for a in p1 for b in p2)
    near(expected_max, max(entropy(p1), entropy(p2)), "Positive-varentropy Jensen equality")
    require(surprise_variance(p1) > 0 and surprise_variance(p2) > 0,
            "Both equality-example varentropies are positive")
    adaptive_gap = 0.9 * (-math.log(0.9) - entropy([0.9, 0.1]))
    require(adaptive_gap < 0, "Adaptive autoregression reverses Jensen comparison")
    return {"positive_varentropy_jensen_gap": expected_max - entropy(p2),
            "adaptive_jensen_gap": adaptive_gap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--max-length", type=int, default=60)
    args = parser.parse_args()
    require(args.max_length >= 1, "max-length must be positive")
    report = {"seed": args.seed, "max_length": args.max_length,
              "geometry": check_tail_geometry(args.seed, args.max_length),
              "length": check_length_counterexamples(), "bounded_model": check_bounded_model(),
              "surprise": check_surprise_decomposition(), "jensen": check_jensen_counterexamples()}
    report["status"] = "PASS"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
