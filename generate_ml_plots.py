import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

def generate_ml_plots():
    # Load data and model
    df = pd.read_csv("titanic/cleaned.csv")
    model = joblib.load("titanic_model.joblib")
    
    # Feature columns (must match model.py)
    feature_cols = [
        "Pclass", "Sex_Val", "AgeFill", "Fare", "FamilySize",
        "Embarked_Val_1", "Embarked_Val_2", "Embarked_Val_3",
        "Title_Val", "Deck_Val", "IsAlone"
    ]
    features_in_df = [col for col in feature_cols if col in df.columns]
    
    X = df[features_in_df]
    y = df["Survived"]
    
    # Predictions
    y_pred = model.predict(X)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead', 'Survived'], yticklabels=['Dead', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Titanic Survival - Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to plots/confusion_matrix.png")
    
    # 2. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=features_in_df).sort_values(ascending=True)
        plt.figure(figsize=(10, 8))
        feat_importances.plot(kind='barh', color='skyblue')
        plt.title('Titanic Survival - Feature Importance')
        plt.xlabel('Relative Importance')
        plt.savefig('plots/feature_importance.png')
        plt.close()
        print("Feature importance plot saved to plots/feature_importance.png")
        print("\nFeature Importances:")
        for name, imp in feat_importances.sort_values(ascending=False).items():
            print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    generate_ml_plots()
