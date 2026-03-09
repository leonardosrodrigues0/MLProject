from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/nsl_kdd_dataset.csv")


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    target_col = df.columns[-1]

    print(f"Dataset: {DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print()

    print("Columns:")
    print(list(df.columns))
    print()

    print("Data types:")
    print(df.dtypes.astype(str).to_string())
    print()

    print("First 5 rows:")
    print(df.head(5).to_string())
    print()

    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(f"Target column: {target_col}")
    print()

    print("Target distribution:")
    print(df[target_col].value_counts().to_string())


if __name__ == "__main__":
    main()
