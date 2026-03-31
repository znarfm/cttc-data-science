import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple


def train_model(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains and compares multiple classifiers on the Titanic dataset.
    Args:
        df: The cleaned Titanic DataFrame.
    Returns:
        tuple: (best_trained_model, metrics_dict)
    """
    # Feature selection (updated with new features)
    feature_cols = [
        "Pclass",
        "Sex_Val",
        "AgeFill",
        "Fare",
        "FamilySize",
        "Embarked_Val_1",
        "Embarked_Val_2",
        "Embarked_Val_3",
        "Title_Val",
        "Deck_Val",
        "IsAlone",
    ]

    # Ensure all columns exist in the DataFrame
    features_in_df = [col for col in feature_cols if col in df.columns]

    X = df[features_in_df]
    y = df["Survived"]

    # Model Pipelines & Param Grids
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Logistic Regression": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
                ]
            ),
            "params": {"classifier__C": [0.1, 1.0, 10.0]},
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
            },
        },
    }

    best_score = 0
    best_model = None
    best_model_name = ""
    comparison_results = {}

    print("Starting model comparison and hyperparameter tuning...")
    for name, config in models.items():
        print(f"  Training {name}...")
        grid = GridSearchCV(config["model"], config["params"], cv=5, scoring="accuracy")
        grid.fit(X, y)

        comparison_results[name] = {
            "best_cv_score": grid.best_score_,
            "best_params": grid.best_params_,
        }

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_model_name = name

    print(f"Best model found: {best_model_name} with CV accuracy: {best_score:.4f}")

    # Final evaluation on a hold-out set for reporting metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Collect metrics
    metrics = {
        "best_model_name": best_model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "model_comparison": comparison_results,
        "feature_names": features_in_df,
    }

    # Feature importance extract
    if hasattr(best_model, "feature_importances_"):
        metrics["feature_importance"] = dict(
            zip(features_in_df, best_model.feature_importances_.tolist())
        )
    elif hasattr(best_model, "steps") and hasattr(best_model.steps[-1][1], "coef_"):
        metrics["feature_importance"] = dict(
            zip(features_in_df, best_model.steps[-1][1].coef_[0].tolist())
        )

    return best_model, metrics


def predict_survival(
    model: Any, input_data: Dict[str, Any], feature_names: list
) -> Tuple[int, float]:
    """
    Predicts survival for a given passenger profile.
    """
    # Create a single-row DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Ensure all features exist (fill missing with 0 for dummy cols)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return int(prediction), float(probability)


if __name__ == "__main__":
    cleaned_path = Path("titanic/cleaned.csv")
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        model, metrics = train_model(df)
        print(
            f"\nFinal Model ({metrics['best_model_name']}) Accuracy: {metrics['accuracy']:.2%}"
        )
        # Save model
        joblib.dump(model, "titanic_model.joblib")
        print("Model saved to titanic_model.joblib")
    else:
        print("Cleaned data not found. Please run cleaning first.")
