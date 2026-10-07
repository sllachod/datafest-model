"""Validate a CatBoost submission for the DataFest competition."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.common import ROOT


def validate_submission(path: Path) -> None:
    """Raise an error when a submission does not match the contest contract."""
    test = pd.read_csv(ROOT / "test.csv")
    submission = pd.read_csv(path)
    expected_columns = ["id_cliente", "prediccion"]

    if list(submission.columns) != expected_columns:
        raise ValueError(
            f"Columnas incorrectas: se esperaban {expected_columns}, "
            f"se encontraron {list(submission.columns)}"
        )
    if len(submission) != len(test):
        raise ValueError(
            f"Cantidad de filas incorrecta: se esperaban {len(test)}, "
            f"se encontraron {len(submission)}"
        )
    if not submission["id_cliente"].equals(test["id_cliente"]):
        raise ValueError("Los id_cliente no coinciden con test.csv o cambiaron de orden")
    if submission["prediccion"].isna().any():
        raise ValueError("La columna prediccion contiene valores faltantes")
    if not submission["prediccion"].between(0, 1).all():
        raise ValueError("Todas las predicciones deben estar entre 0 y 1")

    print(f"Entrega válida: {len(submission):,} predicciones en {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "models" / "catboost" / "submission.csv",
    )
    args = parser.parse_args()
    validate_submission(args.submission)


if __name__ == "__main__":
    main()
