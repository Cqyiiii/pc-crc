#!/usr/bin/env python3
"""Reproduce numerical checks for Propositions 1--2 and the mirror-flow model.

Run from the repository root with python -m theory.verify_rl_theory.

Dependencies: Python 3, NumPy, SciPy. No data files, randomness, network access,
or output files are required. Checks supplement the analytic proofs; they do
not establish claims about the fitted LLMs. A failed check raises an exception.
"""

from itertools import product
import json
import math
from time import perf_counter

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import expit, logit, softmax


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_close(actual, expected, atol=2e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def check_group_identities():
    """Enumerate entire groups independently of the closed-form identities."""
    laws = [
        (0.17, 0.31, 0.52),
        (0.01, 0.30, 0.69),
        (0.40, 0.50, 0.10),
        (0.00, 0.30, 0.70),
        (0.00, 0.00, 1.00),
        (1.00, 0.00, 0.00),
    ]
    # A deliberately non-zero-mean feature checks covariance, not just E[r psi].
    features = np.array([[1.0, 2.0], [-0.3, 0.7], [0.8, -0.4]])
    largest_error = 0.0
    configurations = 0
    for G, probabilities, penalty in product((2, 4, 6), laws, (0, 0.25, 1, 2)):
        p = np.asarray(probabilities)
        reward = np.array([1.0, -penalty, 0.0])
        conditional_numerators = np.zeros(3)
        mean_gradient = np.zeros(2)
        active_probability = 0.0
        failure_only_probability = 0.0
        for outcomes in product(range(3), repeat=G):
            outcomes = np.asarray(outcomes)
            mass = float(np.prod(p[outcomes]))
            if mass == 0:
                continue
            rewards = reward[outcomes]
            advantages = rewards - rewards.mean()
            conditional_numerators[outcomes[0]] += mass * advantages[0]
            mean_gradient += mass * np.mean(
                advantages[:, None] * features[outcomes], axis=0
            )
            active = bool(np.ptp(rewards) > 0)
            active_probability += mass * active
            failure_only_probability += mass * active * np.all(outcomes != 0)
            if penalty == 0:
                require(np.all(advantages[outcomes == 2] <= 0),
                        "A sampled U had positive binary-reward advantage")
        gamma = (G - 1) / G
        mu = p @ reward
        present = p > 0
        observed = conditional_numerators[present] / p[present]
        expected = gamma * (reward[present] - mu)
        covariance = gamma * ((p * reward) @ features - mu * (p @ features))
        active_expected = (
            1 - p[0] ** G - (1 - p[0]) ** G
            if penalty == 0 else 1 - np.sum(p ** G)
        )
        failure_expected = (
            0.0 if penalty == 0
            else (1 - p[0]) ** G - p[1] ** G - p[2] ** G
        )
        for actual, target in [
            (observed, expected),
            (mean_gradient, covariance),
            (active_probability, active_expected),
            (failure_only_probability, failure_expected),
        ]:
            check_close(actual, target)
            largest_error = max(
                largest_error, float(np.max(np.abs(np.asarray(actual) - target)))
            )
        configurations += 1
    return {"configurations": configurations, "max_error": largest_error}


def check_geometry_counterexamples():
    """Check opposite conclusions under explicitly different geometries."""
    p = np.array([0.01, 0.30, 0.69])
    reward = np.array([1.0, -1.0, 0.0])
    mu = p @ reward
    logit_gradient = p * (reward - mu)
    probability_drift = p * (logit_gradient - p @ logit_gradient)
    s2 = p @ p
    correct_drift = (
        p[0] ** 2 * (1 - 2 * p[0] + s2)
        - p[0] * p[1] * (s2 - p[0] - p[1])
    )
    check_close(probability_drift[0], correct_drift)
    check_close(correct_drift, -0.00061398)
    require(correct_drift < 0, "Expected correct-probability drift is not negative")

    p_shared = np.array([0.4, 0.5, 0.1])
    slopes = np.array([4.0, 0.0, 1.0])
    dp_dtheta = p_shared * (slopes - p_shared @ slopes)
    dj_dtheta = dp_dtheta @ reward
    shared_u_drift = dp_dtheta[2] * dj_dtheta
    require(-(p_shared @ reward) > 0, "U conditional-advantage sign changed")
    check_close(shared_u_drift, -0.1239)

    # Integrate the replicator ODE and compare with its independent closed form.
    times = np.linspace(0, 12, 61)
    solutions = {}
    for penalty in (0.0, 1.0):
        r = np.array([1.0, -penalty, 0.0])
        sol = solve_ivp(
            lambda _, state: state * (r - state @ r),
            (0, times[-1]), p, t_eval=times, rtol=2e-11, atol=1e-13,
        )
        require(sol.success, sol.message)
        expected = softmax(np.log(p)[:, None] + r[:, None] * times, axis=0)
        check_close(sol.y, expected, atol=3e-9)
        solutions[penalty] = sol.y
    require(np.all(solutions[1.0][0] >= solutions[0.0][0] - 3e-9),
            "Full-simplex penalty counterexample no longer increases p_R")

    p4 = np.array([0.1, 0.4, 0.2, 0.3])  # R,W,explicit abstention,truncation.
    r4 = np.array([1.0, -1.0, 0.0, 0.0])
    mu4 = p4 @ r4
    euclidean_logit_gradient = p4 * (r4 - mu4)
    check_close(euclidean_logit_gradient[2] - euclidean_logit_gradient[3],
                -mu4 * (p4[2] - p4[3]))
    replicator = p4 * (r4 - mu4)
    check_close(replicator[2] / p4[2] - replicator[3] / p4[3], 0)
    return {"correct_drift": correct_drift, "shared_U_drift": shared_u_drift}


def check_transformed_decoder():
    """A two-step fixed-support decoder exposes the raw-score/Q-score gap."""
    def law(theta):
        first = softmax([theta, 0.0])
        masses, rewards, raw_scores, actual_scores = [], [], [], []
        for prefix in (0, 1):
            raw = softmax([theta, 0.0, -2.0] if prefix == 0
                          else [0.0, 2 * theta, -1.0])
            slopes = np.array([1.0, 0.0, 0.0] if prefix == 0
                              else [0.0, 2.0, 0.0])
            sampled = raw[:2] / raw[:2].sum()
            first_score = float(prefix == 0) - first[0]
            for token in (0, 1):
                masses.append(first[prefix] * sampled[token])
                rewards.append(float(prefix == 1))
                raw_scores.append(first_score + slopes[token] - raw @ slopes)
                actual_scores.append(
                    first_score + slopes[token] - sampled @ slopes[:2]
                )
        return tuple(np.asarray(x) for x in
                     (masses, rewards, raw_scores, actual_scores))

    mass, reward, raw_score, actual_score = law(0.0)
    mu = mass @ reward
    covariance = lambda score: mass @ (reward * score) - mu * (mass @ score)
    step = 1e-5
    positive = law(step)
    negative = law(-step)
    numerical_gradient = (
        positive[0] @ positive[1] - negative[0] @ negative[1]
    ) / (2 * step)
    actual_covariance = covariance(actual_score)
    raw_covariance = covariance(raw_score)
    check_close(actual_covariance, numerical_gradient, atol=1e-9)
    require(abs(raw_covariance - numerical_gradient) > 0.02,
            "The example no longer distinguishes raw and decoder scores")
    return {
        "reward_gradient": numerical_gradient,
        "actual_score_covariance": actual_covariance,
        "raw_score_covariance": raw_covariance,
    }


def check_mirror_flow():
    """Integrate in logits to keep probabilities valid near both boundaries."""
    penalty, kappa, q0 = 1.0, 0.5, 0.2
    qc = penalty / (1 + penalty)
    divergence = qc * math.log(qc / q0) + (1 - qc) * math.log(
        (1 - qc) / (1 - q0)
    )
    critical_a = 1 - math.exp(-divergence / kappa)
    check_close(critical_a, 0.36)
    cases = [
        (0.25, 120.0, "starvation"),
        (0.50, 120.0, "recovery"),
        (critical_a - 0.001, 10000.0, "starvation"),
        (critical_a, 10000.0, "critical"),
        (critical_a + 0.001, 10000.0, "recovery"),
    ]
    results = []
    for a0, horizon, regime in cases:
        def rhs(_, state):
            a, q = expit(state[:2])
            return [
                (1 + penalty) * q - penalty,
                kappa * (1 + penalty) * a,
                a,
                a * q,
            ]

        initial = [logit(a0), logit(q0), 0.0, 0.0]
        sol = solve_ivp(rhs, (0, horizon), initial, dense_output=True,
                        rtol=2e-11, atol=1e-12)
        require(sol.success, sol.message)
        sample = sol.sol(np.linspace(0, horizon, 501))
        u, v = sample[:2]
        a, q = expit(u), expit(v)
        invariant = (
            qc * np.logaddexp(0, -v)
            + (1 - qc) * np.logaddexp(0, v)
            - kappa * np.logaddexp(0, u)
        )
        invariant_error = float(np.max(np.abs(invariant - invariant[0])))
        require(invariant_error < 5e-8, "Mirror-flow invariant failed")
        scale = kappa * (1 + penalty)
        check_close(sample[2, -1], (v[-1] - v[0]) / scale, atol=5e-8)
        check_close(sample[3, -1],
                    (np.logaddexp(0, v[-1]) - np.logaddexp(0, v[0])) / scale,
                    atol=5e-8)
        if regime == "starvation":
            require(divergence > -kappa * math.log1p(-a0),
                    "Incorrect starvation-side classification")
            # Closed form for lambda=1, kappa=.5, q0=.2, derived from invariant.
            q_limit = (1 - math.sqrt(1 - 4 * q0 * (1 - q0) / (1 - a0))) / 2
            require(a[-1] < 1e-8, "Gate did not approach starvation boundary")
            check_close(q[-1], q_limit, atol=2e-8)
            check_close(sample[2, -1], (logit(q_limit) - logit(q0)) / scale,
                        atol=5e-8)
            check_close(sample[3, -1],
                        math.log((1 - q0) / (1 - q_limit)) / scale, atol=5e-8)
        elif regime == "recovery":
            require(divergence < -kappa * math.log1p(-a0),
                    "Incorrect recovery-side classification")
            require(a[-1] > 1 - 1e-7 and q[-1] > 1 - 1e-7,
                    "Flow did not recover")
        else:
            check_close(divergence, -kappa * math.log1p(-a0))
            check_close(a, (1 - 2 * q) ** 2, atol=2e-9)
            check_close(horizon * (qc - q[-1]), 1.0, atol=0.01)
            check_close(horizon ** 2 * a[-1], 4.0, atol=0.03)
        results.append({
            "a0": a0, "regime": regime,
            "a_final": float(a[-1]), "q_final": float(q[-1]),
            "invariant_error": invariant_error,
        })
    # Binary reward recovery uses the same model, with positive initial mass.
    binary = solve_ivp(
        lambda _, state: [expit(state[1]), kappa * expit(state[0])],
        (0, 120), [logit(0.25), logit(q0)], rtol=2e-11, atol=1e-12,
    )
    require(binary.success, binary.message)
    require(np.all(expit(binary.y[:, -1]) > 1 - 1e-7),
            "Binary mirror-flow example did not recover")
    return {"critical_a": critical_a, "cases": results}


def check_bayes_shift():
    source_cutoff = 0.8
    target_cutoff = 1 - math.sqrt(0.02)
    source_wrong = quad(lambda z: 1 - z, source_cutoff, 1)[0]
    transferred_wrong = quad(lambda z: 2 * (1 - z), source_cutoff, 1)[0]
    recalibrated_wrong = quad(lambda z: 2 * (1 - z), target_cutoff, 1)[0]
    check_close([source_wrong, transferred_wrong, recalibrated_wrong],
                [0.02, 0.04, 0.02])
    return {
        "source_wrong": source_wrong,
        "transferred_wrong": transferred_wrong,
        "target_wrong_after_adjustment": recalibrated_wrong,
        "target_dual_multiplier": target_cutoff / (1 - target_cutoff),
    }


def check_common_positive_abstention_reward():
    """A common positive U reward is an affine reward transformation."""
    G = 4
    p = np.array([0.17, 0.31, 0.52])
    gamma = (G - 1) / G
    largest_error = 0.0
    for penalty, rho in product((0.0, 0.25, 1.0), (0.0, 0.2, 0.7, 0.99)):
        reward = np.array([1.0, -penalty, rho])
        effective_penalty = (penalty + rho) / (1 - rho)
        transformed = np.array([1.0, -effective_penalty, 0.0])
        conditional_u = 0.0
        for outcomes in product(range(3), repeat=G):
            outcomes = np.asarray(outcomes)
            r = reward[outcomes]
            rp = transformed[outcomes]
            advantage = r - r.mean()
            scaled_advantage = (1 - rho) * (rp - rp.mean())
            check_close(advantage, scaled_advantage)
            largest_error = max(
                largest_error, float(np.max(abs(advantage - scaled_advantage))))
            if outcomes[0] == 2:
                conditional_u += np.prod(p[outcomes]) * advantage[0] / p[2]
        expected_u = gamma * (rho * (p[0] + p[1]) + penalty * p[1] - p[0])
        check_close(conditional_u, expected_u)
        precision = p[0] / (p[0] + p[1])
        threshold = (penalty + rho) / (1 + penalty)
        require((expected_u > 0) == (precision < threshold),
                "Positive-U phase threshold failed")
        check_close(threshold, effective_penalty / (1 + effective_penalty))
    # The model vector field is scaled by 1-rho after the transformation.
    for penalty, rho, a, q in product(
        (0.0, 0.25, 1.0), (0.0, 0.2, 0.7), (0.2, 0.7), (0.1, 0.6)
    ):
        kappa = 0.5
        effective_penalty = (penalty + rho) / (1 - rho)
        actual = np.array([
            a * (1 - a) * ((1 + penalty) * q - penalty - rho),
            kappa * (1 + penalty) * a * q * (1 - q),
        ])
        transformed = (1 - rho) * np.array([
            a * (1 - a) * ((1 + effective_penalty) * q - effective_penalty),
            kappa * (1 + effective_penalty) * a * q * (1 - q),
        ])
        check_close(actual, transformed)
    return {"reward_configurations": 12, "max_affine_identity_error": largest_error}


def main():
    start = perf_counter()
    checks = [
        ("group identities", check_group_identities),
        ("geometry and abstention counterexamples", check_geometry_counterexamples),
        ("transformed-decoder score distinction", check_transformed_decoder),
        ("mirror-flow barrier, invariant, and exposure", check_mirror_flow),
        ("Bayes distribution-shift example", check_bayes_shift),
        ("common positive non-submission reward", check_common_positive_abstention_reward),
    ]
    for title, check in checks:
        result = check()
        print(f"PASS: {title}")
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    print(f"All RL theory checks passed in {perf_counter() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
