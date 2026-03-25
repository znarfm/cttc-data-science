import pandas as pd
from rich.console import Console
import argparse
from pathlib import Path


def analyze_data(file_path: str | Path) -> None:
    """
    Performs data analysis on the cleaned Titanic dataset.
    """
    df = pd.read_csv(file_path)
    console = Console()

    console.print("--- [bold blue]Titanic Data Analysis[/bold blue] ---")

    # 1. Dataset Overview
    console.print(f"\n[bold]Total passengers:[/bold] {len(df)}")
    console.print(f"[bold]Overall survival rate:[/bold] {df['Survived'].mean():.2%}")

    # 2. Descriptive Statistics
    console.print("\n[bold]Descriptive statistics for key features:[/bold]")
    console.print(df[["AgeFill", "Fare", "FamilySize"]].describe().round(2))

    # 3. Survival Rate by Class
    console.print("\n[bold]Survival rate by Passenger Class:[/bold]")
    pclass_survival = df.groupby("Pclass")["Survived"].mean().round(4)
    console.print(pclass_survival)

    # 4. Survival Rate by Gender
    console.print("\n[bold]Survival rate by Gender:[/bold]")
    gender_survival = df.groupby("Sex")["Survived"].mean().round(4)
    console.print(gender_survival)

    # 5. Survival Rate by Embarkation Port
    console.print("\n[bold]Survival rate by Embarkation Port:[/bold]")
    port_survival = df.groupby("Embarked")["Survived"].mean().round(4)
    console.print(port_survival)

    # 6. Survival Rate by Family Size
    console.print("\n[bold]Survival rate by Family Size:[/bold]")
    family_survival = df.groupby("FamilySize")["Survived"].mean().round(4)
    console.print(family_survival)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Titanic data")
    parser.add_argument(
        "--input", default="titanic/cleaned_train.csv", help="Input cleaned CSV path"
    )
    args = parser.parse_args()
    analyze_data(args.input)
