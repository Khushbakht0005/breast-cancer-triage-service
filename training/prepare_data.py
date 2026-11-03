"""Export the sklearn breast cancer dataset to a CSV file for DVC tracking."""
import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/breast_cancer.csv")
    args = parser.parse_args()

    df = load_breast_cancer(as_frame=True).frame.copy()
    # Feature names use underscores, matching the API contract
    df.columns = [c.replace(" ", "_") for c in df.columns]
    feature_cols = [c for c in df.columns if c != "target"]

    if df.isnull().sum().sum() != 0:
        raise ValueError("Dataset contains null values")
    if len(feature_cols) != 30:
        raise ValueError(f"Expected 30 features, found {len(feature_cols)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {df.shape[0]} rows, {len(feature_cols)} features + target")


if __name__ == "__main__":
    main()
