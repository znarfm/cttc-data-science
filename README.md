# Titanic Data Science Project

This activity is a thorough data science investigation into the Titanic dataset, focusing on data cleaning, exploratory data analysis (EDA), and visualization for Current Trends and Topics in Computing course.

## Project Structure

- `titanic/raw.csv`: Original passenger data.
- `titanic/cleaned.csv`: Preprocessed data ready for analysis.
- `clean_data.py`: Script to handle missing values, map categoricals, and engineer features.
- `analyze_data.py`: Script to generate descriptive statistics and survival rates by class, gender, and port.
- `visualize_data.py`: Script to generate plots and save them to the `plots/` directory.
- `plots/`: Directory containing generated visualizations (Feature distributions, Survival rates).

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed for dependency management.

### Installation

Sync the environment and install dependencies:

```bash
uv sync
```

## Running the Analysis

The analysis follows a pipeline:

1. **Clean Data**:

   ```bash
   uv run python clean_data.py
   ```

2. **Analyze Data**:

   ```bash
   uv run python analyze_data.py
   ```

3. **Visualize Data**:

   ```bash
   uv run python visualize_data.py
   ```

## Key Findings

- **Survival Rate**: Visual distributions highlight significant survival differences based on Passenger Class and Gender.
- **Feature Engineering**: Imputed age based on class/gender groups and calculated family sizes to improve analysis depth.
