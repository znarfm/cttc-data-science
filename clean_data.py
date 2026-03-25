import pandas as pd


def clean_data(input_path, output_path):
    """
    Performs data cleaning and feature engineering on the Titanic dataset.
    """
    df = pd.read_csv(input_path)

    # 1. Sex Mapping
    sexes = sorted(df["Sex"].unique())
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
    df["Sex_Val"] = df["Sex"].map(genders_mapping).astype(int)

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
        [df, pd.get_dummies(df["Embarked_Val"], prefix="Embarked_Val")], axis=1
    )

    # 5. Family Size Feature
    df["FamilySize"] = df["SibSp"] + df["Parch"]

    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    clean_data("titanic/train.csv", "titanic/cleaned_train.csv")
