"""Shared data loading and causal feature engineering for model runners."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "id_cliente"
MONTH = "mes"
TARGET = "objetivo"

CATEGORY_COLUMNS = [
    "ocupacion",
    "region",
    "canal_adquisicion",
    "banda_riesgo",
    "dispositivo_principal",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the competition files from the repository root."""
    return (
        pd.read_csv(ROOT / "train.csv"),
        pd.read_csv(ROOT / "test.csv"),
    )


def _add_causal_history(data: pd.DataFrame) -> pd.DataFrame:
    """Add customer history using only observations from earlier months."""
    result = data.sort_values([ID, MONTH], kind="mergesort").copy()
    grouped = result.groupby(ID, sort=False)
    result["hist_observaciones"] = grouped.cumcount()
    result["hist_conversiones"] = grouped[TARGET].transform(
        lambda values: values.shift().fillna(0).cumsum()
    )
    result["hist_tasa_conversion"] = (
        result["hist_conversiones"]
        / result["hist_observaciones"].replace(0, pd.NA)
    ).fillna(0.0)
    return result


def split_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return causally enriched features, labels, and training months."""
    train_rows = train.copy()
    test_rows = test.copy()
    train_rows["_row_order"] = range(len(train_rows))
    test_rows["_row_order"] = range(len(test_rows))
    train_rows["_is_train"] = True
    test_rows[TARGET] = 0
    test_rows["_is_train"] = False

    combined = _add_causal_history(
        pd.concat([train_rows, test_rows], ignore_index=True)
    )
    enriched_train = combined.loc[combined["_is_train"]].sort_values("_row_order")
    enriched_test = combined.loc[~combined["_is_train"]].sort_values("_row_order")
    drop_columns = [TARGET, "_is_train", "_row_order", ID]

    return (
        enriched_train.drop(columns=drop_columns),
        enriched_test.drop(columns=drop_columns),
        train[TARGET].astype(int),
        train[MONTH],
    )
