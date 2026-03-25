import pandas as pd
from rich import print

def analyze_data(file_path):
    """
    Performs data analysis on the cleaned Titanic dataset.
    """
    df = pd.read_csv(file_path)
    
    print("--- Titanic Data Analysis ---")
    
    # 1. Dataset Overview
    print(f"\nTotal passengers: {len(df)}")
    print(f"Overall survival rate: {df['Survived'].mean():.2%}")
    
    # 2. Descriptive Statistics
    print("\nDescriptive statistics for key features:")
    print(df[['AgeFill', 'Fare', 'FamilySize']].describe())
    
    # 3. Survival Rate by Class
    print("\nSurvival rate by Passenger Class:")
    pclass_survival = df.groupby('Pclass')['Survived'].mean()
    print(pclass_survival)
    
    # 4. Survival Rate by Gender
    print("\nSurvival rate by Gender:")
    gender_survival = df.groupby('Sex')['Survived'].mean()
    print(gender_survival)
    
    # 5. Survival Rate by Embarkation Port
    print("\nSurvival rate by Embarkation Port:")
    port_survival = df.groupby('Embarked')['Survived'].mean()
    print(port_survival)
    
    # 6. Survival Rate by Family Size
    print("\nSurvival rate by Family Size:")
    family_survival = df.groupby('FamilySize')['Survived'].mean()
    print(family_survival)

if __name__ == "__main__":
    analyze_data('titanic/cleaned_train.csv')
