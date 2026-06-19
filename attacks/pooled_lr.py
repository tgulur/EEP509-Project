"""Pooled likelihood-ratio attack on logit-transformed target confidences.

Fits one in-set Gaussian and one out-of-set Gaussian over the target model's
true-label confidences, then scores each sample by the log-likelihood ratio.
Not the per-example LiRA of Carlini et al. 2022 (that's lira_full.py) - this
is closer to a calibrated Yeom variant, sometimes called a global LR attack.
Kept around because it's cheap and outperforms the full LiRA whenever the
reference cache is too small for stable per-sample Gaussian fits.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from attacks.base import MIAttack


class PooledLRAttack(MIAttack):
    """Pooled LR attack. Two Gaussians (in/out) on logit-transformed confidences."""

    def __init__(self, device: torch.device | str = "cpu", eps: float = 1e-6) -> None:
        self.device = torch.device(device)
        self.eps = eps
        self.member_mean = 0.0
        self.member_std = 1.0
        self.non_member_mean = 0.0
        self.non_member_std = 1.0
        self.is_fitted = False

    def fit(
        self,
        model: torch.nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> "PooledLRAttack":
        member_conf = self._true_label_confidence(model, member_loader)
        non_member_conf = self._true_label_confidence(model, non_member_loader)
        self.member_mean = float(member_conf.mean())
        self.member_std = float(member_conf.std() + self.eps)
        self.non_member_mean = float(non_member_conf.mean())
        self.non_member_std = float(non_member_conf.std() + self.eps)
        self.is_fitted = True
        return self

    def score(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit before score.")
        conf = self._true_label_confidence(model, samples)
        # likelihood ratio: P(conf | member) / P(conf | non-member)
        member_logp = _normal_logpdf(conf, self.member_mean, self.member_std)
        non_member_logp = _normal_logpdf(conf, self.non_member_mean, self.non_member_std)
        return member_logp - non_member_logp

    @torch.no_grad()
    def _true_label_confidence(self, model: torch.nn.Module, loader: DataLoader) -> np.ndarray:
        model.to(self.device)
        model.eval()
        values: list[np.ndarray] = []
        for features, labels, _is_member in loader:
            probs = torch.softmax(model(features.to(self.device)), dim=1).cpu()
            conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            values.append(torch.logit(conf.clamp(self.eps, 1.0 - self.eps)).numpy())
        return np.concatenate(values)


def _normal_logpdf(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    var = std**2
    return -0.5 * np.log(2.0 * np.pi * var) - ((values - mean) ** 2) / (2.0 * var)
