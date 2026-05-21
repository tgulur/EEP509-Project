"""Feldman-Zhang memorization scores from the LiRA reference cache.

Definition (Feldman & Zhang 2020, https://arxiv.org/abs/2008.03703):

    mem(x) = P(model correct on x | x IN training) - P(model correct on x | x OUT of training)

Intuition - what mem(x) is actually measuring:
  mem(x) ~= 0   easy/general sample. the model gets it right whether or not x
                was in its training set, so the model learned a *pattern* that
                generalizes to x. nothing to memorize, nothing to leak.
  mem(x) ~= 1   memorized sample. the model is correct on x only when it has
                literally seen x during training, and it didn't learn a rule
                covering x, it learned the row. these are the long-tail patients:
                rare procedures, unusual feature combinations, anything the
                population-level signal can't cover.
  mem(x) < 0    rare/noisy. being in training actively hurts. usually mislabeled.

Estimation: there's no closed form for the probabilities so we use the LiRA
reference models as an empirical influence estimator. each reference model is
trained on a random ~50% subsample of the data, so for any target sample x we
have ~K/2 "in" observations and ~K/2 "out" observations, and mem(x) is just the
difference of mean correctness over those two groups. with K=64 the IN/OUT
samples are stable enough for a quartile-ish stratification of the population.

Why we want this signal: the simple loss-based attack uses loss as its score,
so stratifying samples by loss quartile and then claiming "the attack works
best on the high-loss bin" is partly circular as the attack itself decided which
samples are "high loss." which is flagged in the progress report, and named F/Z as the
natural fix. F/Z scores come from the reference models' argmax *correctness*,
not from the target model's loss, so the stratification is independent of the
attack being evaluated. if mem_high samples also show elevated TPR@FPR under
the attack, that's a real disparate-impact signal rather than a tautology.

Cost: nothing extra. LiRA already trains the reference models. the only change
to the cache was tracking correctness alongside confidence per (model, sample).
"""

from __future__ import annotations

import numpy as np

from attacks.lira_reference import ReferenceModelCache


def compute_feldman_zhang_scores(
    cache: ReferenceModelCache,
    min_models_per_side: int = 2,
) -> dict[int, float]:
    if cache.correctness_matrix.size == 0:
        raise ValueError(
            "ReferenceModelCache has no correctness_matrix. Re-train reference "
            "models with the updated cache so correctness is captured."
        )

    scores: dict[int, float] = {}
    for sample_idx, pos in cache._idx_to_pos.items():
        in_mask = cache.membership_masks[:, pos]
        out_mask = ~in_mask
        if in_mask.sum() < min_models_per_side or out_mask.sum() < min_models_per_side:
            continue
        correct = cache.correctness_matrix[:, pos]
        scores[sample_idx] = float(correct[in_mask].mean() - correct[out_mask].mean())
    return scores
