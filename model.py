import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple


def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Trains a Random Forest Classifier on the Titanic dataset.
    Args:
        df: The cleaned Titanic DataFrame.
    Returns:
        tuple: (trained_model, metrics_dict)
    """
    # Feature selection
    feature_cols = [
        "Pclass",
        "Sex_Val",
        "AgeFill",
        "Fare",
        "FamilySize",
        "Embarked_Val_1",
        "Embarked_Val_2",
        "Embarked_Val_3",
    ]
    
    # Ensure all columns exist in the DataFrame
    features_in_df = [col for col in feature_cols if col in df.columns]
    
    X = df[features_in_df]
    y = df["Survived"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Collect metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(zip(features_in_df, model.feature_importances_.tolist())),
        "feature_names": features_in_df
    }
    
    return model, metrics


def predict_survival(model: RandomForestClassifier, input_data: Dict[str, Any], feature_names: list) -> Tuple[int, float]:
    """
    Predicts survival for a given passenger profile.
    Args:
        model: Trained RandomForest model.
        input_data: Dictionary of feature values.
        feature_names: List of features used during training.
    Returns:
        tuple: (prediction_int, probability_float)
    """
    # Create a single-row DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Ensure features match training columns
    input_df = input_df[feature_names]
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return int(prediction), float(probability)


if __name__ == "__main__":
    # Test script for CLI
    cleaned_path = Path("titanic/cleaned.csv")
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        model, metrics = train_model(df)
        print(f"Model Accuracy: {metrics['accuracy']:.2%}")
        # Save model
        joblib.dump(model, "titanic_model.joblib")
        print("Model saved to titanic_model.joblib")
    else:
        print("Cleaned data not found. Please run cleaning first.")
