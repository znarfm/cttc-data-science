import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_data(file_path, output_dir='plots'):
    """
    Generates visualizations for the Titanic dataset.
    """
    df = pd.read_csv(file_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Global plot styling
    plt.rc('figure', figsize=(10, 5))
    figsize_with_subplots = (10, 10)

    # 1. Feature Grid Visualization
    plt.figure(figsize=figsize_with_subplots) 
    fig_dims = (3, 2)

    # Survival Counts
    plt.subplot2grid(fig_dims, (0, 0))
    df['Survived'].value_counts().plot(kind='bar', title='Survival Counts')

    # Pclass Counts
    plt.subplot2grid(fig_dims, (0, 1))
    df['Pclass'].value_counts().plot(kind='bar', title='Passenger Class Counts')

    # Gender Counts
    plt.subplot2grid(fig_dims, (1, 0))
    df['Sex'].value_counts().plot(kind='bar', title='Gender Counts')
    plt.xticks(rotation=0)

    # Embarked Counts
    plt.subplot2grid(fig_dims, (1, 1))
    df['Embarked'].value_counts().plot(kind='bar', title='Ports of Embarkation Counts')

    # Age Histogram
    plt.subplot2grid(fig_dims, (2, 0))
    df['Age'].hist()
    plt.title('Age Histogram')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png')
    plt.close()

    # 2. Survival Rate by Feature
    # Pclass
    pclass_xt = pd.crosstab(df['Pclass'], df['Survived'])
    pclass_xt_pct = pclass_xt.div(pclass_xt.sum(axis=1).astype(float), axis=0)
    pclass_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Pclass')
    plt.savefig(f'{output_dir}/survival_rate_pclass.png')
    plt.close()

    # Sex
    sex_xt = pd.crosstab(df['Sex_Val'], df['Survived'])
    sex_xt_pct = sex_xt.div(sex_xt.sum(axis=1).astype(float), axis=0)
    sex_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Gender')
    plt.savefig(f'{output_dir}/survival_rate_gender.png')
    plt.close()

    # Age Groups
    df1 = df[df['Survived'] == 0]['AgeFill']
    df2 = df[df['Survived'] == 1]['AgeFill']
    plt.hist([df1, df2], bins=8, stacked=True)
    plt.legend(('Died', 'Survived'), loc='best')
    plt.title('Survivors by Age Groups Histogram')
    plt.savefig(f'{output_dir}/survival_rate_age.png')
    plt.close()

    # Family Size
    df1 = df[df['Survived'] == 0]['FamilySize']
    df2 = df[df['Survived'] == 1]['FamilySize']
    plt.hist([df1, df2], bins=int(df['FamilySize'].max()) + 1, stacked=True)
    plt.legend(('Died', 'Survived'), loc='best')
    plt.title('Survivors by Family Size')
    plt.savefig(f'{output_dir}/survival_rate_family_size.png')
    plt.close()

    print(f"Visualizations saved to directory: {output_dir}")

if __name__ == "__main__":
    visualize_data('titanic/cleaned_train.csv')
