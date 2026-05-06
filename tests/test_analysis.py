import pandas as pd

from analysis.summary import write_kd_transfer_summary


def test_kd_transfer_summary_classifies_reduced_leakage(tmp_path):
    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"model_type": "teacher", "attack_type": "lira", "auc": 0.65, "train_acc": 0.63, "test_acc": 0.47},
            {"model_type": "student", "attack_type": "lira", "auc": 0.51, "train_acc": 0.33, "test_acc": 0.34},
        ]
    ).to_csv(results, index=False)

    output = write_kd_transfer_summary(results, tmp_path)
    frame = pd.read_csv(output)

    assert frame.loc[0, "outcome"] == "reduced"
    assert frame.loc[0, "auc_delta_student_minus_teacher"] < 0
