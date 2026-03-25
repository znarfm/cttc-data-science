import pandas as pd
import argparse
from pathlib import Path


def clean_data(input_path: str | Path, output_path: str | Path) -> None:
    """
    Performs data cleaning and feature engineering on the Titanic dataset.
    """
    df = pd.read_csv(input_path)

    # 1. Sex Mapping
    df["Sex_Val"] = pd.Categorical(df["Sex"]).codes

    # 2. Age Imputation
    # Fill missing Age using the median of their Sex and Pclass group
    df["AgeFill"] = df["Age"]
    df["AgeFill"] = (
        df["AgeFill"]
        .groupby([df["Sex_Val"], df["Pclass"]])
        .transform(lambda x: x.fillna(x.median()))
    )

    # 3. Embarked Mapping
    # Note: Following notebook logic where nan=0, C=1, Q=2, S=3
    embarked_locs = sorted(df["Embarked"].dropna().unique())
    embarked_locs_mapping = {loc: i for i, loc in enumerate(embarked_locs)}
    df["Embarked_Val"] = df["Embarked"].map(embarked_locs_mapping).fillna(0).astype(int)

    # Replace NaN mapping (0) with 'S' mapping (3) as per notebook analysis
    df.loc[df["Embarked_Val"] == 0, "Embarked_Val"] = 3

    # 4. Create Dummy Variables for Embarked
    df = pd.concat(
        [df, pd.get_dummies(df["Embarked_Val"], prefix="Embarked_Val", dtype=int)],
        axis=1,
    )

    # 5. Family Size Feature
    df["FamilySize"] = df["SibSp"] + df["Parch"]

    # Save the cleaned data
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Titanic data")
    parser.add_argument("--input", default="titanic/raw.csv", help="Input CSV path")
    parser.add_argument(
        "--output", default="titanic/cleaned.csv", help="Output CSV path"
    )
    args = parser.parse_args()
    clean_data(args.input, args.output)
