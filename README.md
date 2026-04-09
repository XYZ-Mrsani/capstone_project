# Turbofan Engine Remaining Useful Life (RUL) Prediction System

This project implements an end-to-end predictive maintenance pipeline for estimating the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the **NASA C-MAPSS FD001** dataset. The system includes data preprocessing, feature engineering, exploratory data analysis (EDA), baseline and advanced model training, evaluation, and a Streamlit dashboard for interactive visualization and prediction.

## Project Features

- Loads and processes the NASA C-MAPSS FD001 dataset
- Performs preprocessing and feature engineering on multivariate sensor data
- Generates exploratory data analysis (EDA) visualizations
- Trains and evaluates four prediction models:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LSTM
- Saves trained models and evaluation outputs
- Provides an interactive Streamlit dashboard for model comparison and engine-level analysis

## Project Structure

```text
project-root/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── outputs/
│   ├── eda/
│   ├── models/
│   └── model_comparison.csv
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── baseline_models.py
│   ├── advanced_models.py
│   ├── dashboard.py
│   └── pipeline.py
├── requirements.txt
└── README.md
```

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.10 or higher
- pip (Python package installer)

## Installation

Navigate to the project root directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all required Python libraries for the project.

## Running the Full Pipeline

Run the following command from the project root directory:

```bash
python src/pipeline.py
```

This command will:

1. Load the NASA C-MAPSS FD001 dataset from the `data/` directory.
2. Perform exploratory data analysis (EDA).
3. Preprocess the data and engineer features.
4. Train all four models:
   - Linear Regression
   - Random Forest
   - XGBoost
   - LSTM
5. Save EDA plots to `outputs/eda/`.
6. Save trained models to `outputs/models/`.
7. Save model comparison results to `outputs/model_comparison.csv`.

## Running the Dashboard

After running the pipeline, launch the Streamlit dashboard with:

```bash
python -m streamlit run src/dashboard.py
```

Your browser should automatically open the dashboard. If it does not, open the following URL manually:

```text
http://localhost:8501
```

## Dashboard Pages

The dashboard includes the following sections:

- **Home**  
  View a summary of the project and dataset.

- **Engine Analysis**  
  Select a specific test engine to view its sensor behavior and predicted RUL from each model.

- **Model Comparison**  
  Compare the performance metrics of all trained models side by side.

- **EDA Visualizations**  
  Browse the charts and plots generated during exploratory data analysis.

## Output Files

After a successful run, the following outputs are generated:

- `outputs/eda/`  
  Contains exploratory data analysis visualizations and plots.

- `outputs/models/`  
  Contains the saved trained models.

- `outputs/model_comparison.csv`  
  Contains the evaluation summary for all models.

## Notes

- Make sure the NASA C-MAPSS FD001 dataset files are placed correctly inside the `data/` directory before running the pipeline.
- Run the full pipeline before opening the dashboard so that the required models and output files are available.
- Depending on your hardware, LSTM training may take longer than the other models.

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow / keras
- matplotlib
- seaborn
- streamlit

## Author (258744V)

Project developed as part of a capstone project on predictive maintenance and RUL estimation for turbofan engines.
