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
                    - **AgeFill**: Missing ages imputed using median by Sex and Pclass.
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

            st.subheader("Feature Statistics")
            st.write(results["stats"])

    elif page == "Visualizations":
        st.title("📈 Interactive Exploratory Data Analysis")

        if not cleaned_path.exists():
            st.warning("Please clean the data first.")
        else:
            plots = create_interactive_plots(cleaned_path)

            st.subheader("1. 📊 Survival Distributions")
            st.plotly_chart(plots["feature_distributions"], use_container_width=True)

            st.subheader("2. 🔍 Deep Dive by Features")

            viz_tabs = st.tabs(
                ["Passenger Class", "Gender", "Age Groups", "Family Size"]
            )

            with viz_tabs[0]:
                st.plotly_chart(plots["survival_pclass"], use_container_width=True)
            with viz_tabs[1]:
                st.plotly_chart(plots["survival_gender"], use_container_width=True)
            with viz_tabs[2]:
                if "age_dist" in plots:
                    st.plotly_chart(plots["age_dist"], use_container_width=True)
            with viz_tabs[3]:
                if "family_size" in plots:
                    st.plotly_chart(plots["family_size"], use_container_width=True)

    elif page == "AI/ML Prediction":
        from model import train_model, predict_survival
        import plotly.express as px
        import plotly.graph_objects as go
        import joblib

        st.title("🤖 Survival Prediction (AI/ML)")
        st.markdown(
            "Train a machine learning model using the cleaned data and test custom passenger scenarios."
        )

        model_path = Path("titanic_model.joblib")

        if not cleaned_path.exists():
            st.warning("Please clean the data first.")
        else:
            df = pd.read_csv(cleaned_path)

            # Model Training / Loading Logic
            col_train1, col_train2 = st.columns([2, 1])
            with col_train1:
                if st.button("🚀 Train AI Model", type="primary"):
                    with st.spinner("Training Random Forest Classifier..."):
                        model, metrics = train_model(df)
                        st.session_state.model = model
                        st.session_state.metrics = metrics
                        # Save to disk
                        joblib.dump({"model": model, "metrics": metrics}, model_path)
                        st.success("Model trained and saved successfully!")

            # Load existing model if not in session state
            if "model" not in st.session_state and model_path.exists():
                saved_data = joblib.load(model_path)
                st.session_state.model = saved_data["model"]
                st.session_state.metrics = saved_data["metrics"]
                st.info("Loaded existing model from disk.")

            if "model" in st.session_state:
                # Dashboard for Metrics
                st.markdown("---")
                st.subheader("📊 Model Performance & Insights")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric(
                    "Model Accuracy", f"{st.session_state.metrics['accuracy']:.2%}"
                )

                # Feature Importance Chart
                fi = st.session_state.metrics["feature_importance"]
                fi_df = pd.DataFrame(
                    list(fi.items()), columns=["Feature", "Importance"]
                ).sort_values(by="Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Model Feature Importance",
                    template="plotly_white",
                )
                st.plotly_chart(fig_fi, width="stretch")

                st.markdown("---")
                st.subheader("🔮 Interactive Survival Predictor")
                st.markdown(
                    "Customize a passenger profile to see the predicted survival probability."
                )

                # Prediction Form
                with st.form("prediction_form"):
                    p_col1, p_col2 = st.columns(2)

                    with p_col1:
                        pclass = st.selectbox(
                            "Ticket Class (Pclass)", [1, 2, 3], index=2
                        )
                        sex = st.selectbox("Gender", ["Female", "Male"], index=1)
                        age = st.slider("Age (Years)", 1, 80, 30)
                        fare = st.number_input(
                            "Passenger Fare ($)", min_value=0.0, value=30.0
                        )

                    with p_col2:
                        sibsp = st.number_input(
                            "Number of Siblings/Spouses (SibSp)", 0, 8, 0
                        )
                        parch = st.number_input(
                            "Number of Parents/Children (Parch)", 0, 6, 0
                        )
                        embarked = st.selectbox(
                            "Port of Embarkation",
                            ["Cherbourg", "Queenstown", "Southampton"],
                            index=2,
                        )

                    predict_btn = st.form_submit_button(
                        "🚀 Predict Survival", type="primary"
                    )

                if predict_btn:
                    # Prepare Inputs
                    input_data = {
                        "Pclass": pclass,
                        "Sex_Val": 0 if sex == "Female" else 1,
                        "AgeFill": float(age),
                        "Fare": float(fare),
                        "FamilySize": float(sibsp + parch),
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

                    # Visual Indicator for Probability
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
                st.info("No model found. Click the button above to train the AI.")


if __name__ == "__main__":
    main()
