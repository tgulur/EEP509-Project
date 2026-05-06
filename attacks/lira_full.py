"""Full LiRA (Carlini et al. 2022) with multiple reference models.

The proper way to do LiRA - train N shadow models on random subsets,
then for each target sample compute the likelihood ratio using models
that did/didn't include that sample. Expensive but much more accurate.

Paper: https://arxiv.org/abs/2112.03570

TODO: add support for online LiRA (train reference models incrementally)
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
    """
    Full LiRA - for each sample, we have ~N/2 models that trained on it
    and ~N/2 that didn't. Fit Gaussians to each group's confidence scores,
    then compute log-likelihood ratio.
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

    def fit(
        self,
        model: torch.nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> "FullLiRAAttack":
        """Fit requires reference model cache to be set via set_reference_cache()."""
        if self.reference_cache is None:
            raise RuntimeError(
                "FullLiRAAttack requires reference models. "
                "Call set_reference_cache() before fit()."
            )
        self.is_fitted = True
        return self

    def set_reference_cache(self, cache: ReferenceModelCache) -> None:
        """Set the reference model cache containing trained models and membership info."""
        self.reference_cache = cache

    def score(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        """Compute LiRA scores using reference model confidences."""
        if not self.is_fitted or self.reference_cache is None:
            raise RuntimeError("Call fit() with reference cache before scoring.")

        dataset = samples.dataset
        if not hasattr(dataset, "source_indices"):
            raise ValueError("Dataset must have source_indices for LiRA scoring")

        sample_indices = np.array(dataset.source_indices)
        scores = []

        for idx in sample_indices:
            in_confs, out_confs = self.reference_cache.get_confidences_for_sample(idx)

            if len(in_confs) < self.min_models_per_sample or len(out_confs) < self.min_models_per_sample:
                scores.append(0.0)
                continue

            mu_in = np.mean(in_confs)
            sigma_in = np.std(in_confs) + self.eps
            mu_out = np.mean(out_confs)
            sigma_out = np.std(out_confs) + self.eps

            target_conf = self.reference_cache.get_target_confidence(idx)
            if target_conf is None:
                scores.append(0.0)
                continue

            log_p_in = _normal_logpdf(target_conf, mu_in, sigma_in)
            log_p_out = _normal_logpdf(target_conf, mu_out, sigma_out)
            scores.append(log_p_in - log_p_out)

        return np.array(scores)

    def score_with_target_model(
        self,
        target_model: torch.nn.Module,
        samples: DataLoader,
    ) -> np.ndarray:
        """Compute LiRA scores using the target model's confidence as the query point."""
        if self.reference_cache is None:
            raise RuntimeError("Reference cache not set")

        dataset = samples.dataset
        if not hasattr(dataset, "source_indices"):
            raise ValueError("Dataset must have source_indices")

        sample_indices = np.array(dataset.source_indices)
        target_confs = self._compute_target_confidences(target_model, samples)
        scores = []

        for i, idx in enumerate(sample_indices):
            in_confs, out_confs = self.reference_cache.get_confidences_for_sample(idx)

            if len(in_confs) < self.min_models_per_sample or len(out_confs) < self.min_models_per_sample:
                scores.append(0.0)
                continue

            mu_in = np.mean(in_confs)
            sigma_in = np.std(in_confs) + self.eps
            mu_out = np.mean(out_confs)
            sigma_out = np.std(out_confs) + self.eps

            target_conf = target_confs[i]
            log_p_in = _normal_logpdf(target_conf, mu_in, sigma_in)
            log_p_out = _normal_logpdf(target_conf, mu_out, sigma_out)
            scores.append(log_p_in - log_p_out)

        return np.array(scores)

    @torch.no_grad()
    def _compute_target_confidences(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> np.ndarray:
        """Compute confidence for true label from target model."""
        model.to(self.device)
        model.eval()
        confs = []
        for features, labels, _is_member in loader:
            probs = torch.softmax(model(features.to(self.device)), dim=1).cpu()
            conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            conf = torch.logit(conf.clamp(self.eps, 1.0 - self.eps))
            confs.append(conf.numpy())
        return np.concatenate(confs)


def _normal_logpdf(x: float, mean: float, std: float) -> float:
    """Compute log PDF of normal distribution."""
    var = std ** 2
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
    """Run full LiRA attack and return member/non-member scores.

    Args:
        features: Full feature array
        labels: Full label array
        target_indices: Indices of samples to evaluate (train + test)
        member_indices: Indices that are training members
        target_model: The model to attack
        device: Torch device
        config: Configuration dict with lira settings
        cache_dir: Optional directory to cache reference models

    Returns:
        Tuple of (member_scores, non_member_scores)
    """
    from attacks.lira_reference import ReferenceModelCache, train_reference_models

    lira_config = config.get("lira", {})
    num_models = int(lira_config.get("num_reference_models", 64))
    epochs = int(lira_config.get("reference_model_epochs", 5))
    hidden_dims = list(lira_config.get("reference_model_hidden_dims", [128, 64]))
    sample_fraction = float(lira_config.get("reference_sample_fraction", 0.5))

    cache = train_reference_models(
        features=features,
        labels=labels,
        target_indices=target_indices,
        num_models=num_models,
        hidden_dims=hidden_dims,
        epochs=epochs,
        sample_fraction=sample_fraction,
        device=device,
        num_classes=int(config["model"]["num_classes"]),
        cache_dir=cache_dir,
    )

    cache.compute_target_model_confidences(target_model, features, labels, target_indices, device)

    attack = FullLiRAAttack(device)
    attack.set_reference_cache(cache)
    attack.is_fitted = True

    member_set = set(member_indices)
    member_target_indices = [i for i in target_indices if i in member_set]
    non_member_target_indices = [i for i in target_indices if i not in member_set]

    member_scores = []
    for idx in member_target_indices:
        in_confs, out_confs = cache.get_confidences_for_sample(idx)
        if len(in_confs) < 2 or len(out_confs) < 2:
            member_scores.append(0.0)
            continue
        mu_in, sigma_in = np.mean(in_confs), np.std(in_confs) + 1e-6
        mu_out, sigma_out = np.mean(out_confs), np.std(out_confs) + 1e-6
        target_conf = cache.get_target_confidence(idx)
        if target_conf is None:
            member_scores.append(0.0)
            continue
        member_scores.append(_normal_logpdf(target_conf, mu_in, sigma_in) - _normal_logpdf(target_conf, mu_out, sigma_out))

    non_member_scores = []
    for idx in non_member_target_indices:
        in_confs, out_confs = cache.get_confidences_for_sample(idx)
        if len(in_confs) < 2 or len(out_confs) < 2:
            non_member_scores.append(0.0)
            continue
        mu_in, sigma_in = np.mean(in_confs), np.std(in_confs) + 1e-6
        mu_out, sigma_out = np.mean(out_confs), np.std(out_confs) + 1e-6
        target_conf = cache.get_target_confidence(idx)
        if target_conf is None:
            non_member_scores.append(0.0)
            continue
        non_member_scores.append(_normal_logpdf(target_conf, mu_in, sigma_in) - _normal_logpdf(target_conf, mu_out, sigma_out))

    return np.array(member_scores), np.array(non_member_scores)
