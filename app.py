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
        ["Overview", "Data Cleaning", "Statistical Analysis", "Visualizations", "AI/ML Prediction"],
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
                st.info("Cleaned data already exists. Click the button above to re-run the pipeline.")
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
            col1.metric("Overall Survival Rate", f"{results['overall_survival_rate']:.2%}")
            col2.metric("Cleaned Dataset Size", results['total_passengers'])
            
            st.subheader("Survival Rates by Feature")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("**By Gender**")
                st.table(results['gender_survival'])
                
                st.markdown("**By Passenger Class**")
                st.table(results['pclass_survival'])
                
            with res_col2:
                st.markdown("**By Family Size**")
                st.table(results['family_survival'])
                
                if 'port_survival' in results:
                    st.markdown("**By Port**")
                    st.table(results['port_survival'])

            st.subheader("Feature Statistics")
            st.write(results['stats'])

    elif page == "Visualizations":
        st.title("📈 Interactive Exploratory Data Analysis")
        
        if not cleaned_path.exists():
            st.warning("Please clean the data first.")
        else:
            plots = create_interactive_plots(cleaned_path)
            
            st.subheader("1. 📊 Survival Distributions")
            st.plotly_chart(plots["feature_distributions"], use_container_width=True)
            
            st.subheader("2. 🔍 Deep Dive by Features")
            
            viz_tabs = st.tabs(["Passenger Class", "Gender", "Age Groups", "Family Size"])
            
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
        st.title("🤖 Survival Prediction (AI/ML)")
        st.markdown(
            """
            > [!NOTE]
            > **Planned Feature**: Interact with the model by inputting passenger data to predict survival probability.
            """
        )
        
        st.info("The machine learning model integration is currently under development.")
        
        # Premium Placeholder UI
        with st.container():
            st.markdown("### 🛠️ Development Progress")
            st.progress(25)
            
            st.subheader("Upcoming Features:")
            st.markdown(
                """
                - [ ] Scikit-learn Model Training (Random Forest/XGBoost)
                - [ ] Interactive Input Form for custom predictions
                - [ ] Model Performance Metrics (ROC/AUC, Confusion Matrix)
                - [ ] Feature Importance Visualization
                """
            )
            
            if st.button("Check Model Compatibility"):
                try:
                    import sklearn
                    st.success(f"Scikit-learn version {sklearn.__version__} is ready!")
                except ImportError:
                    st.error("Scikit-learn not found. Use 'uv add scikit-learn' to install.")

if __name__ == "__main__":
    main()
