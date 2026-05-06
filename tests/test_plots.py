import pandas as pd

from evaluation.plots import make_all_plots


def test_privacy_utility_plot_uses_numeric_utility(tmp_path):
    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "model_type": "teacher",
                "attack_type": "loss_based",
                "mitigation": "none",
                "seed": 509,
                "auc": 0.61,
                "tpr_at_01fpr": 0.01,
                "tpr_at_05fpr": 0.02,
                "tpr_at_1fpr": 0.03,
                "train_acc": 0.62,
                "test_acc": 0.46,
            }
        ]
    ).to_csv(results, index=False)

    figure_dir = tmp_path / "figures"
    make_all_plots(results, figure_dir)

    assert (figure_dir / "privacy_utility_tradeoff.png").exists()
    assert (figure_dir / "privacy_utility_tradeoff.pdf").exists()
