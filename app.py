import streamlit as st
import pandas as pd
from pathlib import Path
from clean_data import clean_data
from analyze_data import analyze_data
from visualize_data import create_interactive_plots

# Page Config
st.set_page_config(
    page_title="Titanic Data Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data(path="titanic/raw.csv"):
    if Path(path).exists():
        return pd.read_csv(path)
    return None


def main():
    st.sidebar.title("🚢 Titanic Explorer")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Data Cleaning",
            "Statistical Analysis",
            "Visualizations",
            "AI/ML Prediction",
        ],
    )

    raw_data = load_data()
    cleaned_path = Path("titanic/cleaned.csv")

    if raw_data is None:
        st.error("Raw data file 'titanic/raw.csv' not found.")
        return

    if page == "Overview":
        st.title("🚢 Titanic Dataset Overview")
        st.markdown(
            """
            Welcome to the **Titanic Data Science Explorer**. This dashboard allows you to explore, 
            clean, and analyze the famous Titanic passenger dataset.
            """
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Passengers", len(raw_data))
        col2.metric("Features", len(raw_data.columns))
        col3.metric("Missing Values", raw_data.isna().sum().sum())

        st.subheader("Raw Data Preview")
        st.dataframe(raw_data, width="stretch")

        with st.expander("Feature Legend"):
            st.markdown(
                """
                - **Survived**: 0 = No, 1 = Yes
                - **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
                - **Sex**: Passenger gender
                - **Age**: Age in years
                - **SibSp**: # of siblings / spouses aboard
                - **Parch**: # of parents / children aboard
                - **Ticket**: Ticket number
                - **Fare**: Passenger fare
                - **Cabin**: Cabin number
                - **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
                """
            )

    elif page == "Data Cleaning":
        st.title("🧹 Data Cleaning & Preprocessing")
        st.markdown(
            "Clean the data by imputing missing values and mapping categorical features to numerical values."
        )

        if st.button("🚀 Run Cleaning Pipeline", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned_df = clean_data("titanic/raw.csv", "titanic/cleaned.csv")
                st.success("Data cleaned and saved to 'titanic/cleaned.csv'!")

                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_df, width="stretch")

                st.subheader("Transformation Summary")
                st.markdown(
                    """
                    - **Title Extraction**: Mapping names to social titles (Mr, Miss, Mrs, Master, Rare).
                    - **Deck Extraction**: Extracting deck levels (A-G, T) from cabin numbers; mapped 'M' for missing.
                    - **Refined Age Imputation**: Missing ages imputed using median by **Title** and **Pclass**.
                    - **Fare Handling**: Missing fares filled with Pclass-specific median.
                    - **IsAlone**: New feature indicating if a passenger is traveling without family.
                    - **Sex_Val**: Categorical Sex mapped to codes.
                    - **Embarked_Val/Dummies**: Embarked port mapped and one-hot encoded.
                    - **FamilySize**: Combined SibSp and Parch.
                    """
                )
        else:
            if cleaned_path.exists():
                st.info(
                    "Cleaned data already exists. Click the button above to re-run the pipeline."
                )
                st.dataframe(pd.read_csv(cleaned_path), width="stretch")
            else:
                st.warning("Cleaned data not found. Please run the pipeline.")

    elif page == "Statistical Analysis":
        st.title("📊 Statistical Analysis")

        if not cleaned_path.exists():
            st.warning("Please clean the data first.")
        else:
            results = analyze_data(cleaned_path, silent=True)

            st.subheader("Key Insights")
            col1, col2 = st.columns(2)
            col1.metric(
                "Overall Survival Rate", f"{results['overall_survival_rate']:.2%}"
            )
            col2.metric("Cleaned Dataset Size", results["total_passengers"])

            # Statistical Significance
            st.subheader("Statistical Significance (Age vs Survival)")
            ttest = results.get("age_ttest")
            if ttest:
                t_col1, t_col2 = st.columns(2)
                t_col1.metric("T-statistic", f"{ttest['t_stat']:.4f}")
                p_val = ttest["p_val"]
                is_significant = p_val < 0.05
                t_col2.metric(
                    "P-value",
                    f"{p_val:.4f}",
                    delta="Significant" if is_significant else "Not Significant",
                    delta_color="normal" if is_significant else "inverse",
                )

                if is_significant:
                    st.success(
                        "The difference in age between survivors and non-survivals is statistically significant."
                    )
                else:
                    st.info(
                        "The difference in age is not statistically significant at the 5% level."
                    )

            st.subheader("Survival Rates by Feature")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("**By Gender**")
                st.table(results["gender_survival"])

                st.markdown("**By Passenger Class**")
                st.table(results["pclass_survival"])

            with res_col2:
                st.markdown("**By Family Size**")
                st.table(results["family_survival"])

                if "port_survival" in results:
                    st.markdown("**By Port**")
                    st.table(results["port_survival"])

            st.subheader("Feature Correlation with Survival")
            st.dataframe(results["correlations"], width="stretch")

    elif page == "Visualizations":
        st.title("📈 Interactive Exploratory Data Analysis")

        if not cleaned_path.exists():
            st.warning("Please clean the data first.")
        else:
            plots = create_interactive_plots(cleaned_path)

            viz_tabs = st.tabs(
                [
                    "Survival Counts",
                    "Correlation Heatmap",
                    "Age & Fare Distributions",
                    "Passenger Class",
                    "Multivariate Analysis",
                ]
            )

            with viz_tabs[0]:
                st.plotly_chart(plots["feature_distributions"], width="stretch")
            with viz_tabs[1]:
                st.plotly_chart(plots["correlation_heatmap"], width="stretch")
            with viz_tabs[2]:
                st.plotly_chart(plots["age_violin"], width="stretch")
            with viz_tabs[3]:
                st.plotly_chart(plots["survival_pclass"], width="stretch")
            with viz_tabs[4]:
                st.plotly_chart(plots["fare_age_scatter"], width="stretch")

    elif page == "AI/ML Prediction":
        from model import train_model, predict_survival
        import plotly.express as px
        import plotly.graph_objects as go
        import joblib

        st.title("🤖 Survival Prediction (AI/ML)")
        st.markdown(
            "Train and compare multiple ML models to find the best predictor for survival."
        )

        model_path = Path("titanic_model.joblib")

        if not cleaned_path.exists():
            st.warning("Please clean the data first.")
        else:
            df = pd.read_csv(cleaned_path)

            # Model Training Logic
            if st.button("🚀 Train & Compare Models", type="primary"):
                with st.spinner(
                    "Comparing Random Forest, Logistic Regression, and XGBoost..."
                ):
                    model, metrics = train_model(df)
                    st.session_state.model = model
                    st.session_state.metrics = metrics
                    joblib.dump({"model": model, "metrics": metrics}, model_path)
                    st.success("Models trained and optimized!")

            # Load existing model
            if "model" not in st.session_state and model_path.exists():
                try:
                    saved_data = joblib.load(model_path)
                    st.session_state.model = saved_data["model"]
                    st.session_state.metrics = saved_data["metrics"]
                except Exception:
                    st.warning("Saved model incompatible. Please retrain.")

            if "model" in st.session_state:
                st.markdown("---")
                st.subheader("📊 Model Performance & Insights")

                comp_col1, comp_col2 = st.columns([1, 2])
                with comp_col1:
                    st.metric("Best Model", st.session_state.metrics["best_model_name"])
                    st.metric(
                        "Test Accuracy", f"{st.session_state.metrics['accuracy']:.2%}"
                    )

                with comp_col2:
                    st.markdown("**Comparison Results (CV Accuracy)**")
                    comp_df = pd.DataFrame(
                        st.session_state.metrics["model_comparison"]
                    ).T
                    st.dataframe(comp_df, width="stretch")

                # Feature Importance
                if "feature_importance" in st.session_state.metrics:
                    fi = st.session_state.metrics["feature_importance"]
                    fi_df = pd.DataFrame(
                        list(fi.items()), columns=["Feature", "Importance"]
                    ).sort_values(by="Importance", ascending=True)

                    fig_fi = px.bar(
                        fi_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Feature Significance ({st.session_state.metrics['best_model_name']})",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_fi, width="stretch")

                st.markdown("---")
                st.subheader("🔮 Interactive Survival Predictor")

                # Prediction Form
                with st.form("prediction_form"):
                    p_col1, p_col2 = st.columns(2)

                    with p_col1:
                        pclass = st.selectbox(
                            "Ticket Class (Pclass)", [1, 2, 3], index=2
                        )
                        sex = st.selectbox("Gender", ["Female", "Male"], index=1)
                        age = st.slider("Age (Years)", 1, 80, 30)
                        title = st.selectbox(
                            "Title", ["Mr", "Miss", "Mrs", "Master", "Rare"], index=0
                        )

                    with p_col2:
                        sibsp = st.number_input("Siblings/Spouses (SibSp)", 0, 8, 0)
                        parch = st.number_input("Parents/Children (Parch)", 0, 6, 0)
                        fare = st.number_input(
                            "Passenger Fare ($)", min_value=0.0, value=30.0
                        )
                        deck = st.selectbox(
                            "Deck (from Cabin)",
                            ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"],
                            index=8,
                        )
                        embarked = st.selectbox(
                            "Port",
                            ["Cherbourg", "Queenstown", "Southampton"],
                            index=2,
                        )

                    predict_btn = st.form_submit_button(
                        "🚀 Predict Survival", type="primary"
                    )

                if predict_btn:
                    # Mapping for Deck (Hardcoded based on clean_data.py logic for raw.csv)
                    deck_map = {
                        "A": 0,
                        "B": 1,
                        "C": 2,
                        "D": 3,
                        "E": 4,
                        "F": 5,
                        "G": 6,
                        "Unknown": 7,
                        "T": 8,
                    }

                    input_data = {
                        "Pclass": pclass,
                        "Sex_Val": 0 if sex == "Female" else 1,
                        "AgeFill": float(age),
                        "Fare": float(fare),
                        "FamilySize": float(sibsp + parch),
                        "IsAlone": 1 if (sibsp + parch) == 0 else 0,
                        "Title_Val": {
                            "Mr": 1,
                            "Miss": 2,
                            "Mrs": 3,
                            "Master": 4,
                            "Rare": 5,
                        }[title],
                        "Deck_Val": deck_map[deck],
                        "Embarked_Val_1": 0,
                        "Embarked_Val_2": 0,
                        "Embarked_Val_3": 0,
                    }

                    port_id = {"Cherbourg": 1, "Queenstown": 2, "Southampton": 3}[
                        embarked
                    ]
                    input_data[f"Embarked_Val_{port_id}"] = 1

                    prediction, probability = predict_survival(
                        st.session_state.model,
                        input_data,
                        st.session_state.metrics["feature_names"],
                    )

                    st.markdown("### Prediction Result")
                    if prediction == 1:
                        st.success(
                            f"**Predicted Status**: Survived (Probability: {probability:.2%})"
                        )
                    else:
                        st.error(
                            f"**Predicted Status**: Died (Probability: {1 - probability:.2%})"
                        )

                    fig_prob = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            title={"text": "Survival Probability (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#1e40af"},
                                "steps": [
                                    {"range": [0, 50], "color": "#fecaca"},
                                    {"range": [50, 100], "color": "#bbf7d0"},
                                ],
                            },
                        )
                    )
                    st.plotly_chart(fig_prob, width="stretch")
            else:
                st.info(
                    "No model found. Click the button above to train and compare models."
                )


if __name__ == "__main__":
    main()
