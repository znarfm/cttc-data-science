import pandas as pd
import joblib
from model import train_model
from pathlib import Path

def get_comparison_metrics():
    df = pd.read_csv("titanic/cleaned.csv")
    model, metrics = train_model(df)
    
    print("\nComparison Table Data:")
    print(f"{'Model':<20} | {'CV Accuracy':<15}")
    print("-" * 40)
    for name, result in metrics['model_comparison'].items():
        print(f"{name:<20} | {result['best_cv_score']:.4%}")
    
    print(f"\nFinal Test Accuracy (Best Model - {metrics['best_model_name']}): {metrics['accuracy']:.4%}")

if __name__ == "__main__":
    get_comparison_metrics()
