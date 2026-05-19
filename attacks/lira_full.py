"""Full LiRA (Carlini et al. 2022) with multiple reference models.

The first version of this attack (see attacks/lira.py) fit two Gaussians
on the target model's own confidence scores. That collapses the per-sample
IN/OUT distributions into one population distribution and leaves most of
the LiRA signal on the table - re-reading section 4 of the paper made that
obvious. This file is the real thing: N reference models trained on random
~50% subsets, and for each target sample we fit IN/OUT Gaussians from the
models that did/didn't see it.

Paper: https://arxiv.org/abs/2112.03570
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from attacks.base import MIAttack

if TYPE_CHECKING:
    from attacks.lira_reference import ReferenceModelCache


class FullLiRAAttack(MIAttack):
    """For each sample: fit IN-Gaussian over models that trained on it, OUT-Gaussian
    over models that didn't, then score = log p_in(target_conf) - log p_out(target_conf).
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        eps: float = 1e-6,
        min_models_per_sample: int = 2,
    ) -> None:
        self.device = torch.device(device)
        self.eps = eps
        self.min_models_per_sample = min_models_per_sample
        self.reference_cache: ReferenceModelCache | None = None
        self.is_fitted = False

    def set_reference_cache(self, cache: ReferenceModelCache) -> None:
        self.reference_cache = cache

    def fit(
        self,
        model: torch.nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> "FullLiRAAttack":
        if self.reference_cache is None:
            raise RuntimeError(
                "FullLiRAAttack requires reference models. "
                "Call set_reference_cache() before fit()."
            )
        self.is_fitted = True
        return self

    def score(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        if not self.is_fitted or self.reference_cache is None:
            raise RuntimeError("Call fit() with reference cache before scoring.")
        sample_indices = _require_source_indices(samples)
        target_confs = [self.reference_cache.get_target_confidence(idx) for idx in sample_indices]
        return self._score_indices(sample_indices, target_confs)

    def score_with_target_model(
        self,
        target_model: torch.nn.Module,
        samples: DataLoader,
    ) -> np.ndarray:
        """Score using the live target model rather than the cached confidences."""
        if self.reference_cache is None:
            raise RuntimeError("Reference cache not set")
        sample_indices = _require_source_indices(samples)
        target_confs = self._compute_target_confidences(target_model, samples)
        return self._score_indices(sample_indices, target_confs)

    def _score_indices(self, sample_indices: np.ndarray, target_confs) -> np.ndarray:
        cache = self.reference_cache
        min_n = self.min_models_per_sample
        scores = np.zeros(len(sample_indices), dtype=np.float64)
        for i, idx in enumerate(sample_indices):
            in_confs, out_confs = cache.get_confidences_for_sample(idx)
            if len(in_confs) < min_n or len(out_confs) < min_n:
                continue
            target_conf = target_confs[i]
            if target_conf is None:
                continue
            scores[i] = _likelihood_ratio(target_conf, in_confs, out_confs, self.eps)
        return scores

    @torch.no_grad()
    def _compute_target_confidences(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> np.ndarray:
        model.to(self.device)
        model.eval()
        confs = []
        for features, labels, _is_member in loader:
            probs = torch.softmax(model(features.to(self.device)), dim=1).cpu()
            conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            confs.append(torch.logit(conf.clamp(self.eps, 1.0 - self.eps)).numpy())
        return np.concatenate(confs)


def _require_source_indices(samples: DataLoader) -> np.ndarray:
    dataset = samples.dataset
    if not hasattr(dataset, "source_indices"):
        raise ValueError("Dataset must have source_indices for LiRA scoring")
    return np.array(dataset.source_indices)


def _likelihood_ratio(target_conf: float, in_confs: np.ndarray, out_confs: np.ndarray, eps: float) -> float:
    mu_in, sigma_in = float(np.mean(in_confs)), float(np.std(in_confs)) + eps
    mu_out, sigma_out = float(np.mean(out_confs)), float(np.std(out_confs)) + eps
    return _normal_logpdf(target_conf, mu_in, sigma_in) - _normal_logpdf(target_conf, mu_out, sigma_out)


def _normal_logpdf(x: float, mean: float, std: float) -> float:
    var = std * std
    return -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)


def run_full_lira(
    features: np.ndarray,
    labels: np.ndarray,
    target_indices: np.ndarray,
    member_indices: np.ndarray,
    target_model: torch.nn.Module,
    device: torch.device,
    config: dict,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train N reference models, score every target index, split member/non-member."""
    from attacks.lira_reference import train_reference_models

    lira_config = config.get("lira", {})
    cache = train_reference_models(
        features=features,
        labels=labels,
        target_indices=target_indices,
        num_models=int(lira_config.get("num_reference_models", 64)),
        hidden_dims=list(lira_config.get("reference_model_hidden_dims", [128, 64])),
        epochs=int(lira_config.get("reference_model_epochs", 5)),
        sample_fraction=float(lira_config.get("reference_sample_fraction", 0.5)),
        device=device,
        num_classes=int(config["model"]["num_classes"]),
        cache_dir=cache_dir,
    )
    cache.compute_target_model_confidences(target_model, features, labels, target_indices, device)

    member_set = set(int(i) for i in member_indices)
    member_scores: list[float] = []
    non_member_scores: list[float] = []

    for idx in target_indices:
        in_confs, out_confs = cache.get_confidences_for_sample(idx)
        target_conf = cache.get_target_confidence(idx)
        if len(in_confs) < 2 or len(out_confs) < 2 or target_conf is None:
            score = 0.0
        else:
            score = _likelihood_ratio(target_conf, in_confs, out_confs, eps=1e-6)
        (member_scores if int(idx) in member_set else non_member_scores).append(score)

    return np.array(member_scores), np.array(non_member_scores)
