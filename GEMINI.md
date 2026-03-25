# GEMINI.md - Project Context

## Project Overview

This project is a data science exploration focused on the Titanic dataset. The primary goal is to perform thorough data cleaning, exploratory data analysis (EDA), and insightful visualizations to understand the characteristics of the Titanic passengers.

- **Main Technologies:** Python, Jupyter Notebook, `pandas`, `numpy`, `matplotlib`, `seaborn`.
- **Dependency Management:** `uv` is used for managing Python versions and dependencies.
- **Architecture:** The project is centered around a single Jupyter Notebook (`titanic.ipynb`) which follows a data analysis pipeline:
  1. Setup and Data Loading
  2. Data Cleaning and Preprocessing (handling missing values, type conversions)
  3. Exploratory Data Analysis (EDA)
  4. Data Visualization (univariate and multivariate analysis)

## Building and Running

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed on your system.

### Installation

Sync the environment and install dependencies:

```bash
uv sync
```

### Running the Project

To run the analysis, open the Jupyter Notebook:

```bash
# Using jupyter (if installed in the environment)
uv run jupyter notebook titanic.ipynb
```

Alternatively, use any IDE with Jupyter support (like VS Code or PyCharm) pointing to the `.venv` created by `uv`.

## Development Conventions

- **Exploratory Workflow:** All analysis and visualization are performed within `titanic.ipynb`.
- **Code Quality:** Prioritize clear, documented code for data transformations and use descriptive labels for all visualizations.
- **Validation:** Data integrity is verified through summary statistics and visual inspection of distributions.
