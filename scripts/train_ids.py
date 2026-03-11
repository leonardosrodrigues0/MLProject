"""
Loads the NSL-KDD CSV, performs basic cleaning and feature preparation, splits the
full dataset into train/validation/test, then trains a baseline classifier.

Run:
  python3 scripts/train_ids.py --drop-duplicates --drop-constant-cols
  python3 scripts/train_ids.py --drop-duplicates --drop-constant-cols --verbose
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


DEFAULT_DATASET_PATH = Path("data/nsl_kdd_dataset.csv")
TARGET_COL = "labels"
DEFAULT_CATEGORICAL_COLS = ("protocol_type", "service", "flag")
DEFAULT_THRESHOLD = 0.50


def _parse_hidden_layer_sizes(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("hidden-layer-sizes must be a comma-separated list, e.g. 64,32")
    try:
        sizes = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("hidden-layer-sizes must contain integers") from e
    if any(s <= 0 for s in sizes):
        raise argparse.ArgumentTypeError("hidden-layer-sizes must be positive integers")
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--drop-duplicates", action="store_true")
    parser.add_argument(
        "--drop-duplicates-after-labelmap",
        action="store_true",
        help="Drop duplicates after mapping labels to binary (dedupe on features + binary target).",
    )
    parser.add_argument(
        "--drop-cols",
        type=str,
        default="",
        help="Comma-separated list of raw input columns to drop before encoding/scaling.",
    )
    parser.add_argument(
        "--drop-constant-cols",
        action="store_true",
        help="Drop columns with <= 1 unique value in the full dataset.",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--hidden-layer-sizes", type=_parse_hidden_layer_sizes, default=(64, 32))
    parser.add_argument("--alpha", type=float, default=0.00005)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--n-iter-no-change", type=int, default=10)
    parser.add_argument("--learning-rate-init", type=float, default=0.001)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold for predicting attack (class=1) from predicted probability.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print dataset/category diagnostics (may be long for high-cardinality columns).",
    )
    return parser.parse_args()


def _drop_constant_columns(df: pd.DataFrame, *, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique(dropna=False) <= 1:
            dropped.append(col)
    if not dropped:
        return df, []
    return df.drop(columns=dropped), dropped


def _split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train-size + val-size + test-size must equal 1.0")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )
    # Split the remaining data into validation and test sets.
    tmp_size = val_size + test_size
    val_fraction_of_tmp = val_size / tmp_size if tmp_size else 0.0
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        train_size=val_fraction_of_tmp,
        random_state=random_state,
        stratify=y_tmp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _print_metrics(split_name: str, y_true: pd.Series, y_pred: pd.Series) -> None:
    # `zero_division=0` avoids noisy warnings when a split has no predicted positives.
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{split_name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")


def _mlp_param_count(input_features: int, hidden_layer_sizes: Sequence[int]) -> int:
    layer_sizes = (input_features, *hidden_layer_sizes, 1)
    total = 0
    for n_in, n_out in zip(layer_sizes, layer_sizes[1:]):
        total += n_in * n_out + n_out
    return int(total)


def _predict_with_threshold(pipeline: Pipeline, X: pd.DataFrame, *, threshold: float) -> pd.Series:
    if not (0.0 < threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1 (exclusive).")

    proba = pipeline.predict_proba(X)
    classes = pipeline.named_steps["model"].classes_
    try:
        attack_idx = list(classes).index(1)
    except ValueError as e:
        raise RuntimeError(f"Expected model classes to include 1, got {classes!r}") from e

    return pd.Series((proba[:, attack_idx] >= threshold).astype("int64"), index=X.index)


def main() -> None:
    args = _parse_args()

    df = pd.read_csv(args.dataset)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")

    if args.drop_duplicates:
        df = df.drop_duplicates()

    if args.drop_constant_cols:
        df, dropped_constant = _drop_constant_columns(df, target_col=TARGET_COL)
    else:
        dropped_constant = []

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    if drop_cols:
        missing = [c for c in drop_cols if c not in df.columns]
        if missing:
            raise ValueError(f"--drop-cols contains unknown columns: {missing}")
        df = df.drop(columns=drop_cols)

    # Binary target: normal vs attack (everything else).
    y = (df[TARGET_COL] != "normal").astype("int64")
    X = df.drop(columns=[TARGET_COL])

    if args.drop_duplicates_after_labelmap:
        before = len(X)
        full = pd.concat([X, y.rename("_y")], axis=1).drop_duplicates()
        X = full.drop(columns=["_y"])
        y = full["_y"].astype("int64")
        removed = before - len(X)
        if removed:
            print(f"Dropped duplicates after label mapping: {removed}")

    categorical_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    if args.verbose:
        print("Categorical column diagnostics (unique values):")
        for col in categorical_cols:
            series = X[col]
            # Keep NaN as an explicit category if it exists (should not for this dataset).
            uniques = series.drop_duplicates()
            unique_values = [
                (v if pd.notna(v) else "<NA>") for v in uniques.tolist()
            ]
            # Stable ordering for readability.
            unique_values_sorted = sorted(unique_values, key=lambda s: str(s))
            print(f"- {col}: {len(unique_values_sorted)} values")
            print(f"  {unique_values_sorted}")
        print()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(
        X,
        y,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols),
        ],
        remainder="drop",
    )
    model = MLPClassifier(
        hidden_layer_sizes=args.hidden_layer_sizes,
        max_iter=args.max_iter,
        alpha=args.alpha,
        early_stopping=args.early_stopping,
        n_iter_no_change=args.n_iter_no_change,
        learning_rate_init=args.learning_rate_init,
        random_state=args.random_state,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    print(f"Dataset: {args.dataset}")
    print(f"Records: {len(df)}  Features (raw): {X.shape[1]}")
    if dropped_constant:
        print(f"Dropped constant columns: {dropped_constant}")
    if drop_cols:
        print(f"Dropped user-selected columns: {drop_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print()

    pipeline.fit(X_train, y_train)

    # Report the dimensionality after encoding/scaling so "model size" is concrete.
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    n_features = int(len(feature_names))
    print(f"Features (after encoding/scaling): {n_features}")
    print(f"MLP hidden layers: {args.hidden_layer_sizes}")
    print(f"Approx. trainable parameters: {_mlp_param_count(n_features, args.hidden_layer_sizes)}")
    print(f"Decision threshold (attack): {args.threshold:.2f}")
    print()

    _print_metrics("Train", y_train, _predict_with_threshold(pipeline, X_train, threshold=args.threshold))
    _print_metrics("Val  ", y_val, _predict_with_threshold(pipeline, X_val, threshold=args.threshold))
    _print_metrics("Test ", y_test, _predict_with_threshold(pipeline, X_test, threshold=args.threshold))


if __name__ == "__main__":
    main()
