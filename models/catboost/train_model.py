"""Train a temporal conversion model and create submission.csv.

Usage:
    python -m models.catboost.train_model
    python -m models.catboost.train_model --iterations 1200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from models.common import CATEGORY_COLUMNS, ID, MONTH, ROOT, load_data, split_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "catboost" / "submission.csv",
    )
    args = parser.parse_args()

    train, test = load_data()
    train_x, test_x, train_y, months = split_features(train, test)
    categorical = [column for column in CATEGORY_COLUMNS if column in train_x]

    validation_month = months.max()
    fit_mask = months < validation_month
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=args.iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=5,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        train_x.loc[fit_mask],
        train_y.loc[fit_mask],
        cat_features=categorical,
        eval_set=(train_x.loc[~fit_mask], train_y.loc[~fit_mask]),
        early_stopping_rounds=80,
        verbose=100,
    )
    validation_pred = model.predict_proba(train_x.loc[~fit_mask])[:, 1]
    auc = roc_auc_score(train_y.loc[~fit_mask], validation_pred)
    print(f"Temporal validation month={validation_month}: AUC={auc:.6f}, Gini={2 * auc - 1:.6f}")

    best_iterations = max(model.get_best_iteration() + 1, 100)
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=best_iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=5,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    final_model.fit(train_x, train_y, cat_features=categorical, verbose=100)
    predictions = np.clip(final_model.predict_proba(test_x)[:, 1], 0, 1)
    submission = pd.DataFrame({ID: test[ID], "prediccion": predictions})
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
