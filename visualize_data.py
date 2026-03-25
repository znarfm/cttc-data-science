import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def visualize_data(file_path: str | Path, output_dir: str | Path = "plots") -> None:
    """
    Generates visualizations for the Titanic dataset using Seaborn.
    """
    df = pd.read_csv(file_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Map raw values to descriptive labels
    df["Survived_Label"] = df["Survived"].map({0: "Died", 1: "Survived"})
    df["Sex"] = df["Sex"].str.title()
    df["Pclass_Label"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    df["Embarked"] = df["Embarked"].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
    
    pclass_order = ["1st Class", "2nd Class", "3rd Class"]

    # Global plot styling
    sns.set_theme(style="whitegrid")

    # 1. Feature Grid Visualization
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

    # Survival Counts
    sns.countplot(data=df, x="Survived_Label", ax=axes[0, 0], order=["Died", "Survived"])
    axes[0, 0].set_title("Survival Counts")
    axes[0, 0].set_xlabel("Survival Status")

    # Pclass Counts
    sns.countplot(data=df, x="Pclass_Label", ax=axes[0, 1], order=pclass_order)
    axes[0, 1].set_title("Passenger Class Counts")
    axes[0, 1].set_xlabel("Passenger Class")

    # Gender Counts
    sns.countplot(data=df, x="Sex", ax=axes[1, 0])
    axes[1, 0].set_title("Gender Counts")

    # Embarked Counts
    sns.countplot(data=df, x="Embarked", ax=axes[1, 1])
    axes[1, 1].set_title("Ports of Embarkation Counts")

    # Age Histogram
    sns.histplot(data=df, x="Age", bins=20, ax=axes[2, 0])
    axes[2, 0].set_title("Age Histogram")

    # Remove empty subplot
    fig.delaxes(axes[2, 1])

    plt.tight_layout()
    plt.savefig(out_path / "feature_distributions.png", dpi=300)
    plt.close()

    # 2. Survival Rate by Feature
    # Pclass
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Pclass_Label", y="Survived", errorbar=None, order=pclass_order)
    plt.title("Survival Rate by Pclass")
    plt.xlabel("Passenger Class")
    plt.savefig(out_path / "survival_rate_pclass.png", dpi=300)
    plt.close()

    # Sex
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Sex", y="Survived", errorbar=None)
    plt.title("Survival Rate by Gender")
    plt.savefig(out_path / "survival_rate_gender.png", dpi=300)
    plt.close()

    # Age Groups
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="AgeFill", hue="Survived_Label", multiple="stack", bins=8)
    plt.title("Survivors by Age Groups Histogram")
    plt.savefig(out_path / "survival_rate_age.png", dpi=300)
    plt.close()

    # Family Size
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=df,
        x="FamilySize",
        hue="Survived_Label",
        multiple="stack",
        bins=int(df["FamilySize"].max()) + 1,
    )
    plt.title("Survivors by Family Size")
    plt.savefig(out_path / "survival_rate_family_size.png", dpi=300)
    plt.close()

    print(f"Visualizations saved to directory: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Titanic data")
    parser.add_argument(
        "--input", default="titanic/cleaned_train.csv", help="Input cleaned CSV path"
    )
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()
    visualize_data(args.input, args.output_dir)
