import pandas as pd
from rich.console import Console
import argparse
from pathlib import Path
from typing import Any, Dict
from scipy import stats


def analyze_data(file_path: str | Path, silent: bool = False) -> Dict[str, Any]:
    """
    Performs data analysis on the cleaned Titanic dataset.
    Returns:
        Dict[str, Any]: A dictionary containing analysis results.
    """
    df = pd.read_csv(file_path)
    console = Console()

    results = {}

    if not silent:
        console.print("--- [bold blue]Titanic Data Analysis[/bold blue] ---")

    # 1. Dataset Overview
    results["total_passengers"] = len(df)
    results["overall_survival_rate"] = df["Survived"].mean()
    if not silent:
        console.print(f"\n[bold]Total passengers:[/bold] {results['total_passengers']}")
        console.print(
            f"[bold]Overall survival rate:[/bold] {results['overall_survival_rate']:.2%}"
        )

    # 2. Descriptive Statistics
    results["stats"] = df[["AgeFill", "Fare", "FamilySize"]].describe().round(2)
    if not silent:
        console.print("\n[bold]Descriptive statistics for key features:[/bold]")
        console.print(results["stats"])

    # 3. Survival Rate by Class
    results["pclass_survival"] = df.groupby("Pclass")["Survived"].mean().round(4)
    if not silent:
        console.print("\n[bold]Survival rate by Passenger Class:[/bold]")
        console.print(results["pclass_survival"])

    # 4. Survival Rate by Gender
    results["gender_survival"] = df.groupby("Sex")["Survived"].mean().round(4)
    if not silent:
        console.print("\n[bold]Survival rate by Gender:[/bold]")
        console.print(results["gender_survival"])

    # 5. Survival Rate by Embarkation Port
    if "Embarked" in df.columns:
        results["port_survival"] = df.groupby("Embarked")["Survived"].mean().round(4)
        if not silent:
            console.print("\n[bold]Survival rate by Embarkation Port:[/bold]")
            console.print(results["port_survival"])

    # 6. Survival Rate by Family Size
    results["family_survival"] = df.groupby("FamilySize")["Survived"].mean().round(4)
    if not silent:
        console.print("\n[bold]Survival rate by Family Size:[/bold]")
        console.print(results["family_survival"])

    # 7. Correlation Analysis
    numerical_df = df.select_dtypes(include=["number"])
    results["correlations"] = numerical_df.corr()["Survived"].sort_values(
        ascending=False
    )
    if not silent:
        console.print("\n[bold]Correlations with Survival:[/bold]")
        console.print(results["correlations"])

    # 8. Statistical Significance (T-test for Age)
    if "AgeFill" in df.columns:
        survived_age = df[df["Survived"] == 1]["AgeFill"].dropna()
        died_age = df[df["Survived"] == 0]["AgeFill"].dropna()
        t_stat, p_val = stats.ttest_ind(survived_age, died_age)
        results["age_ttest"] = {"t_stat": t_stat, "p_val": p_val}
        if not silent:
            console.print(
                "\n[bold]T-test for Age difference (Survivors vs Died):[/bold]"
            )
            console.print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
            if p_val < 0.05:
                console.print(
                    "[green]The difference in age is statistically significant.[/green]"
                )
            else:
                console.print(
                    "[yellow]The difference in age is not statistically significant.[/yellow]"
                )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Titanic data")
    parser.add_argument(
        "--input", default="titanic/cleaned.csv", help="Input cleaned CSV path"
    )
    args = parser.parse_args()
    analyze_data(args.input)
