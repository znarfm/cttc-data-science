import pandas as pd
import argparse
from pathlib import Path
from typing import Optional


def clean_data(
    input_data: str | Path | pd.DataFrame, output_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Performs data cleaning and feature engineering on the Titanic dataset.
    Args:
        input_data: Path to raw CSV or a pandas DataFrame.
        output_path: Optional path to save the cleaned CSV.
    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    if isinstance(input_data, (str, Path)):
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()

    # 1. Sex Mapping
    df["Sex_Val"] = pd.Categorical(df["Sex"]).codes

    # 1a. Title Extraction
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        # Group rare titles
        rare_titles = [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ]
        df["Title"] = df["Title"].replace(rare_titles, "Rare")
        df["Title"] = df["Title"].replace("Mlle", "Miss")
        df["Title"] = df["Title"].replace("Ms", "Miss")
        df["Title"] = df["Title"].replace("Mme", "Mrs")

        # Title Mapping
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df["Title_Val"] = df["Title"].map(title_mapping).fillna(0).astype(int)

    # 1b. Deck Extraction (from Cabin)
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].str[0].fillna("M")
        deck_list = sorted(df["Deck"].unique())
        deck_mapping = {deck: i for i, deck in enumerate(deck_list)}
        df["Deck_Val"] = df["Deck"].map(deck_mapping).astype(int)

    # 2. Age Imputation (Refined using Title and Pclass)
    if "Age" in df.columns:
        df["AgeFill"] = df["Age"]
        group_cols = (
            ["Title_Val", "Pclass"]
            if "Title_Val" in df.columns
            else ["Sex_Val", "Pclass"]
        )
        df["AgeFill"] = df.groupby(group_cols)["AgeFill"].transform(
            lambda x: x.fillna(x.median())
        )

    # 2b. Fare Handling
    if "Fare" in df.columns:
        df["Fare"] = df.groupby("Pclass")["Fare"].transform(
            lambda x: x.fillna(x.median())
        )

    # 3. Embarked Mapping
    # Note: Following notebook logic where nan=0, C=1, Q=2, S=3
    if "Embarked" in df.columns:
        embarked_locs = sorted(df["Embarked"].dropna().unique())
        embarked_locs_mapping = {loc: i for i, loc in enumerate(embarked_locs)}
        df["Embarked_Val"] = (
            df["Embarked"].map(embarked_locs_mapping).fillna(0).astype(int)
        )

        # Replace NaN mapping (0) with 'S' mapping (3) as per notebook analysis
        df.loc[df["Embarked_Val"] == 0, "Embarked_Val"] = 3

        # 4. Create Dummy Variables for Embarked
        df = pd.concat(
            [df, pd.get_dummies(df["Embarked_Val"], prefix="Embarked_Val", dtype=int)],
            axis=1,
        )

    # 5. Family Size Feature
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"]
        df["IsAlone"] = (df["FamilySize"] == 0).astype(int)

    # Save the cleaned data if output_path is provided
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Cleaned data saved to: {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Titanic data")
    parser.add_argument("--input", default="titanic/raw.csv", help="Input CSV path")
    parser.add_argument(
        "--output", default="titanic/cleaned.csv", help="Output CSV path"
    )
    args = parser.parse_args()
    clean_data(args.input, args.output)
