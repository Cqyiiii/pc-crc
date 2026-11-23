from math import inf
from tail_crc import crc_threshold, tail10_nll

# One CAL family per row, all from the same frozen solver and seed.
cal = [("W", 0.3), ("R", 0.1), ("W", 0.7), ("U", inf)]
cut = crc_threshold(cal, "0.4")  # j = floor(5 * 0.4) = 2

# Sampled answer-span log probabilities after the first close and before EOS.
# Remove prefill and injected delimiters/newlines before calling tail10_nll.
for logp, valid in [([-0.1, -0.2], True), ([-0.7], True), ([], False)]:
    score = tail10_nll(logp, valid=valid)
    submit = valid and score < cut
    print(f"score={score:g} cutoff={cut:g} submit={submit}")

assert cut == 0.7 and not (0.7 < cut)  # reject cutoff ties
assert crc_threshold(cal, "0.1") == -inf  # j = 0
assert crc_threshold(cal, "0.8") == inf   # fewer than j wrong answers
