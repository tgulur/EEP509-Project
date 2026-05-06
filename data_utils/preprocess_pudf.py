"""Preprocess raw Texas PUDF tab files into the Texas-100X model CSV."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def build_texas100x_csv(
    data_root: str | Path,
    raw_dir: str = "raw",
    raw_glob: str = "PUDF_base*q*_tab.txt",
    output_file: str = "texas_100x.csv",
    label_column: str = "PRINC_SURG_PROC_CODE",
    feature_columns: Iterable[str] | None = None,
    top_classes: int = 100,
) -> Path:
    """Create Texas-100X processed artifacts from quarterly tab PUDF files.

    This follows the public Texas-100X preprocessing recipe while adapting paths
    and outputs to this repo. The CSV keeps `THCIC_ID` for auditability; the
    downstream loader removes it before model training.
    """

    root = Path(data_root)
    raw_root = root / raw_dir
    raw_files = sorted(raw_root.glob(raw_glob))
    if not raw_files:
        raise FileNotFoundError(f"No raw PUDF tab files matched {raw_root / raw_glob}.")

    columns = list(feature_columns or _default_feature_columns())
    required_columns = columns + [label_column]
    frames: list[pd.DataFrame] = []
    for raw_file in raw_files:
        frame = pd.read_csv(
            raw_file,
            sep="\t",
            dtype=str,
            usecols=lambda column: column in required_columns,
            low_memory=False,
        )
        missing = set(required_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"{raw_file} is missing required columns: {sorted(missing)}")
        frames.append(frame.loc[:, required_columns])

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(inplace=True)

    cat_attrs = [
        "SEX_CODE",
        "TYPE_OF_ADMISSION",
        "SOURCE_OF_ADMISSION",
        "PAT_STATUS",
        "RACE",
        "ETHNICITY",
        "ADMITTING_DIAGNOSIS",
        label_column,
    ]
    combined.loc[combined["SEX_CODE"] == "M", "SEX_CODE"] = 0
    combined.loc[combined["SEX_CODE"] == "F", "SEX_CODE"] = 1
    combined.loc[combined["SEX_CODE"] == "U", "SEX_CODE"] = pd.NA

    for column in cat_attrs:
        combined = combined[pd.to_numeric(combined[column], errors="coerce").notnull()]
        combined[column] = combined[column].astype("int64")

    combined.dropna(inplace=True)
    for column in ["LENGTH_OF_STAY", "PAT_AGE"]:
        combined[column] = combined[column].astype("int64")
    combined["TOTAL_CHARGES"] = combined["TOTAL_CHARGES"].astype("float64")

    top_labels = [
        surgery for surgery, _count in Counter(combined[label_column]).most_common(top_classes)
    ]
    combined = combined[combined[label_column].isin(top_labels)].copy()
    combined[label_column] = combined[label_column].astype("int64")

    mapping: dict[str, dict[int, int]] = {}
    for column in cat_attrs:
        mapping[column] = {value: idx for idx, value in enumerate(sorted(Counter(combined[column]).keys()))}
        combined[column] = combined[column].map(mapping[column]).astype("int64")

    attribute_dict = _build_attribute_dict(combined, cat_attrs, label_column, mapping)
    attribute_idx = {
        column: combined.columns.get_loc(column)
        for column in set(cat_attrs).union({"THCIC_ID", "TOTAL_CHARGES"}) - {label_column}
    }

    output = root / output_file
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    _write_pickle_artifacts(root, combined, label_column, attribute_idx, attribute_dict)
    return output


def _default_feature_columns() -> list[str]:
    return [
        "THCIC_ID",
        "SEX_CODE",
        "TYPE_OF_ADMISSION",
        "SOURCE_OF_ADMISSION",
        "LENGTH_OF_STAY",
        "PAT_AGE",
        "PAT_STATUS",
        "RACE",
        "ETHNICITY",
        "TOTAL_CHARGES",
        "ADMITTING_DIAGNOSIS",
    ]


def _build_attribute_dict(
    frame: pd.DataFrame,
    cat_attrs: list[str],
    label_column: str,
    mapping: dict[str, dict[int, int]],
) -> dict[int, dict[int, str]]:
    attribute_dict: dict[int, dict[int, str]] = {}
    for column in cat_attrs:
        if column == "SEX_CODE":
            attribute_dict[frame.columns.get_loc(column)] = {0: "Male", 1: "Female"}
        elif column == "ETHNICITY":
            attribute_dict[frame.columns.get_loc(column)] = {0: "Hispanic", 1: "Not Hispanic"}
        elif column == "RACE":
            attribute_dict[frame.columns.get_loc(column)] = {
                0: "Native American",
                1: "Asian",
                2: "Black",
                3: "White",
                4: "Other",
            }
        elif column == label_column:
            continue
        else:
            attribute_dict[frame.columns.get_loc(column)] = {
                value: str(value) for value in mapping[column].values()
            }
    return attribute_dict


def _write_pickle_artifacts(
    root: Path,
    frame: pd.DataFrame,
    label_column: str,
    attribute_idx: dict[str, int],
    attribute_dict: dict[int, dict[int, str]],
) -> None:
    y = np.array(frame[label_column])
    x = np.asarray(frame.drop(columns=label_column), dtype=np.float64)
    max_attr_vals = np.max(x, axis=0)
    safe_max = np.where(max_attr_vals == 0, 1.0, max_attr_vals)
    normalized_x = x / safe_max

    with (root / "texas_100x_features.p").open("wb") as handle:
        pickle.dump(normalized_x, handle)
    with (root / "texas_100x_labels.p").open("wb") as handle:
        pickle.dump(y, handle)
    with (root / "texas_100x_feature_desc.p").open("wb") as handle:
        pickle.dump([attribute_idx, attribute_dict, max_attr_vals], handle)
