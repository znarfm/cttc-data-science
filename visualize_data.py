import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any


def create_interactive_plots(file_path: str | Path) -> Dict[str, Any]:
    """
    Generates interactive Plotly visualizations.
    Returns:
        Dict[str, Any]: A dictionary of Plotly figures.
    """
    df = pd.read_csv(file_path)

    # Descriptive Labels
    df["Survived_Label"] = df["Survived"].map({0: "Died", 1: "Survived"})
    df["Sex"] = df["Sex"].str.title()
    df["Pclass_Label"] = df["Pclass"].map(
        {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
    )
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map(
            {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
        )

    plots = {}

    # 1. Main Distribution (Grid replacement)
    plots["feature_distributions"] = px.histogram(
        df,
        x="Survived_Label",
        color="Survived_Label",
        title="Survival Count Distribution",
        labels={"Survived_Label": "Status"},
        template="plotly_white",
    )

    # 2. Survival by Pclass (Aggregated)
    pclass_survival = df.groupby("Pclass_Label")["Survived"].mean().reset_index()
    plots["survival_pclass"] = px.bar(
        pclass_survival,
        x="Pclass_Label",
        y="Survived",
        color="Pclass_Label",
        title="Survival Rate by Passenger Class",
        labels={"Pclass_Label": "Class", "Survived": "Survival Rate"},
        category_orders={"Pclass_Label": ["1st Class", "2nd Class", "3rd Class"]},
        template="plotly_white",
    )

    # 3. Survival by Gender (Aggregated)
    gender_survival = df.groupby("Sex")["Survived"].mean().reset_index()
    plots["survival_gender"] = px.bar(
        gender_survival,
        x="Sex",
        y="Survived",
        color="Sex",
        title="Survival Rate by Gender",
        labels={"Sex": "Gender", "Survived": "Survival Rate"},
        template="plotly_white",
    )

    # 4. Age Distribution
    if "Age" in df.columns:
        plots["age_dist"] = px.histogram(
            df,
            x="Age",
            color="Survived_Label",
            nbins=20,
            title="Age Distribution by Survival Status",
            labels={"Survived_Label": "Status"},
            template="plotly_white",
            barmode="overlay",
        )

    # 5. Family Size
    if "FamilySize" in df.columns:
        plots["family_size"] = px.histogram(
            df,
            x="FamilySize",
            color="Survived_Label",
            title="Survival by Family Size",
            labels={"Survived_Label": "Status"},
            template="plotly_white",
            barmode="group",
        )

    return plots


def visualize_data(
    file_path: str | Path,
    output_dir: Optional[str | Path] = "plots",
    return_figs: bool = False,
) -> Optional[List[plt.Figure]]:
    """
    Generates static visualizations for the Titanic dataset using Seaborn.
    """
    df = pd.read_csv(file_path)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    # Map raw values to descriptive labels
    df["Survived_Label"] = df["Survived"].map({0: "Died", 1: "Survived"})
    df["Sex"] = df["Sex"].str.title()
    df["Pclass_Label"] = df["Pclass"].map(
        {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
    )
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map(
            {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
        )

    pclass_order = ["1st Class", "2nd Class", "3rd Class"]
    figs = []

    # Global plot styling
    sns.set_theme(style="whitegrid")

    # 1. Feature Grid Visualization
    fig1, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

    # Survival Counts
    sns.countplot(
        data=df, x="Survived_Label", ax=axes[0, 0], order=["Died", "Survived"]
    )
    axes[0, 0].set_title("Survival Counts")

    # Pclass Counts
    sns.countplot(data=df, x="Pclass_Label", ax=axes[0, 1], order=pclass_order)
    axes[0, 1].set_title("Passenger Class Counts")

    # Gender Counts
    sns.countplot(data=df, x="Sex", ax=axes[1, 0])
    axes[1, 0].set_title("Gender Counts")

    # Embarked Counts
    if "Embarked" in df.columns:
        sns.countplot(data=df, x="Embarked", ax=axes[1, 1])
        axes[1, 1].set_title("Ports of Embarkation Counts")

    # Age Histogram
    if "Age" in df.columns:
        sns.histplot(data=df, x="Age", bins=20, ax=axes[2, 0])
        axes[2, 0].set_title("Age Histogram")

    fig1.delaxes(axes[2, 1])
    plt.tight_layout()

    if output_dir:
        fig1.savefig(Path(output_dir) / "feature_distributions.png", dpi=300)
    if return_figs:
        figs.append(fig1)
    else:
        plt.close(fig1)

    # Bar Charts (Aggregated)
    for col, title, out in [
        ("Pclass_Label", "Survival Rate by Pclass", "survival_rate_pclass.png"),
        ("Sex", "Survival Rate by Gender", "survival_rate_gender.png"),
    ]:
        fig = plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x=col, y="Survived", errorbar=None)
        plt.title(title)
        if output_dir:
            plt.savefig(Path(output_dir) / out, dpi=300)
        if return_figs:
            figs.append(fig)
        else:
            plt.close(fig)

    return figs if return_figs else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Titanic data")
    parser.add_argument(
        "--input", default="titanic/cleaned.csv", help="Input cleaned CSV path"
    )
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()
    visualize_data(args.input, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Titanic data")
    parser.add_argument(
        "--input", default="titanic/cleaned.csv", help="Input cleaned CSV path"
    )
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()
    visualize_data(args.input, args.output_dir)
