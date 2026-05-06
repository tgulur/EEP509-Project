import numpy as np

from evaluation.metrics import attack_metrics, tpr_at_fpr


def test_attack_metrics_detect_perfect_scores():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    metrics = attack_metrics(y_true, scores)

    assert metrics["auc"] == 1.0
    assert tpr_at_fpr(y_true, scores, 0.01) == 1.0
