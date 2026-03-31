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

    # 1. Main Distribution
    plots["feature_distributions"] = px.histogram(
        df,
        x="Survived_Label",
        color="Survived_Label",
        title="Survival Count Distribution",
        labels={"Survived_Label": "Status"},
        template="plotly_white",
    )

    # 2. Survival by Pclass
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

    # 3. Correlation Heatmap
    numerical_df = df.select_dtypes(include=["number"])
    corr = numerical_df.corr()
    plots["correlation_heatmap"] = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        template="plotly_white",
    )

    # 4. Age Distribution (Violin)
    plots["age_violin"] = px.violin(
        df,
        y="AgeFill",
        x="Survived_Label",
        color="Survived_Label",
        box=True,
        points="all",
        title="Age Distribution by Survival Status",
        labels={"AgeFill": "Age", "Survived_Label": "Status"},
        template="plotly_white",
    )

    # 5. Fare vs Age Scatter
    plots["fare_age_scatter"] = px.scatter(
        df,
        x="AgeFill",
        y="Fare",
        color="Survived_Label",
        size="Pclass",
        hover_data=["Sex", "Pclass"],
        title="Fare vs Age Colored by Survival",
        labels={"AgeFill": "Age"},
        template="plotly_white",
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
    fig1, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

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
    if "AgeFill" in df.columns:
        sns.histplot(data=df, x="AgeFill", bins=20, ax=axes[2, 0], kde=True)
        axes[2, 0].set_title("Age Distribution (Imputed)")

    # Fare Histogram
    if "Fare" in df.columns:
        sns.histplot(data=df, x="Fare", bins=20, ax=axes[2, 1], kde=True)
        axes[2, 1].set_title("Fare Distribution")

    plt.tight_layout()

    if output_dir:
        fig1.savefig(Path(output_dir) / "feature_distributions.png", dpi=300)
    if return_figs:
        figs.append(fig1)

    # 2. Correlation Heatmap
    fig2 = plt.figure(figsize=(12, 10))
    numerical_df = df.select_dtypes(include=["number"])
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    if output_dir:
        plt.savefig(Path(output_dir) / "correlation_heatmap.png", dpi=300)
    if return_figs:
        figs.append(fig2)

    # 3. Violin Plots for Age and Fare
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.violinplot(data=df, x="Survived_Label", y="AgeFill", ax=ax1, split=True)
    ax1.set_title("Age Distribution by Survival")
    sns.violinplot(data=df, x="Survived_Label", y="Fare", ax=ax2, split=True)
    ax2.set_title("Fare Distribution by Survival")
    if output_dir:
        plt.savefig(Path(output_dir) / "violin_plots.png", dpi=300)
    if return_figs:
        figs.append(fig3)

    return figs if return_figs else None

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
