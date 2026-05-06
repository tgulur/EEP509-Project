import pandas as pd
import pytest

pytest.importorskip("torch")

from data_utils.texas100x import Texas100XDataset, load_texas100x_arrays


def test_loader_excludes_thcic_id(tmp_path):
    data_root = tmp_path / "texas100x"
    data_root.mkdir()
    frame = pd.DataFrame(
        {
            "THCIC_ID": [10, 20, 30],
            "AGE": [1.0, 2.0, 3.0],
            "RISK": [0.2, 0.3, 0.4],
            "PRINC_SURG_PROC_CODE": ["A", "B", "A"],
        }
    )
    frame.to_csv(data_root / "texas_100x.csv", index=False)

    features, labels = load_texas100x_arrays(data_root)

    assert features.shape == (3, 2)
    assert labels.tolist() == [0, 1, 0]
    assert 10.0 not in features[:, 0]


def test_dataset_returns_membership_tuple():
    features = [[0.1, 0.2], [0.3, 0.4]]
    labels = [1, 0]
    dataset = Texas100XDataset(features, labels, member_indices=[0], indices=[0, 1])

    sample = dataset[0]
    assert len(sample) == 3
    assert bool(sample[2]) is True
