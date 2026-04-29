# Turbofan Engine Remaining Useful Life (RUL) Prediction System

Final submission for Capstone Project **258744V**.

This project implements an end-to-end predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the **NASA C-MAPSS** turbofan engine degradation simulation dataset. The system includes data loading, preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, prediction output generation, and a Streamlit dashboard for interactive visualization.

## Project Overview

Aircraft turbofan engines generate multivariate time-series sensor data throughout their operating life. This project uses that data to predict how many cycles remain before an engine reaches failure condition. The main objective is to support predictive maintenance by estimating engine RUL accurately and comparing multiple machine learning and deep learning models.

## Key Features

- Loads NASA C-MAPSS turbofan degradation datasets.
- Supports FD001, FD002, FD003, and FD004 dataset files.
- Performs preprocessing and feature engineering on sensor data.
- Removes low-variance / near-constant sensors where required.
- Generates exploratory data analysis outputs.
- Trains and evaluates multiple RUL prediction models.
- Saves trained models and feature column information.
- Generates model comparison results.
- Generates test prediction outputs.
- Provides a Streamlit dashboard for model and prediction visualization.

## Dataset

The project uses the NASA C-MAPSS turbofan engine degradation simulation dataset.

The `data/` folder contains:

```text
data/
├── 6. Turbofan Engine Degradation Simulation...
├── Damage Propagation Modeling.pdf
├── nasa_cmapss.zip
├── readme.txt
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

## Project Structure

```text
Capstone Project 258744V/
├── data/
│   ├── 6. Turbofan Engine Degradation Simulation...
│   ├── Damage Propagation Modeling.pdf
│   ├── nasa_cmapss.zip
│   ├── readme.txt
│   ├── train_FD001.txt
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD001.txt
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   ├── test_FD004.txt
│   ├── RUL_FD001.txt
│   ├── RUL_FD002.txt
│   ├── RUL_FD003.txt
│   └── RUL_FD004.txt
│
├── outputs/
│   ├── eda/
│   ├── models/
│   ├── feature_cols.json
│   ├── model_comparison.csv
│   ├── system_architecture.png
│   └── test_predictions.csv
│
├── src/
│   ├── dashboard.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── models.py
│   └── pipeline.py
│
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- TensorFlow / Keras
- matplotlib
- seaborn
- Streamlit
- joblib

## Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd "Capstone Project 258744V"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

For Windows:

```bash
venv\Scripts\activate
```

For macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

## Running the Full Pipeline

Run the following command from the project root directory:

```bash
python src/pipeline.py
```

The pipeline performs the following steps:

1. Loads the NASA C-MAPSS dataset files from the `data/` folder.
2. Performs data preprocessing and feature engineering.
3. Generates exploratory data analysis outputs.
4. Trains the RUL prediction models.
5. Evaluates model performance using regression metrics.
6. Saves trained models inside `outputs/models/`.
7. Saves selected feature columns to `outputs/feature_cols.json`.
8. Saves model comparison results to `outputs/model_comparison.csv`.
9. Saves test predictions to `outputs/test_predictions.csv`.

## Running the Dashboard

After running the full pipeline, launch the Streamlit dashboard:

```bash
python -m streamlit run src/dashboard.py
```

If the browser does not open automatically, open:

```text
http://localhost:8501
```

## Output Files

After a successful run, the project generates the following outputs:

```text
outputs/
├── eda/
├── models/
├── feature_cols.json
├── model_comparison.csv
├── system_architecture.png
└── test_predictions.csv
```

### Output Description

| Output | Description |
|---|---|
| `outputs/eda/` | Contains exploratory data analysis charts and visualizations. |
| `outputs/models/` | Contains trained and saved machine learning / deep learning models. |
| `outputs/feature_cols.json` | Stores the final selected feature columns used by the models. |
| `outputs/model_comparison.csv` | Stores the evaluation summary of all trained models. |
| `outputs/test_predictions.csv` | Stores predicted RUL values for test engines. |
| `outputs/system_architecture.png` | Shows the system architecture diagram used in the project. |

## Model Evaluation

The system compares model performance using standard RUL prediction evaluation metrics such as:

- RMSE
- MAE
- R² Score
- NASA asymmetric scoring function, where applicable

The model comparison results are saved in:

```text
outputs/model_comparison.csv
```

## Dashboard Functionality

The Streamlit dashboard provides an interactive interface to review the project outputs. It can be used to:

- View overall project summary.
- Explore generated EDA visualizations.
- Compare model evaluation results.
- Review predicted RUL values.
- Analyse model outputs using saved result files.

## Notes

- Ensure all required dataset files are available inside the `data/` folder before running the pipeline.
- Run `python src/pipeline.py` before opening the dashboard.
- The dashboard depends on files generated inside the `outputs/` folder.
- LSTM/deep learning training may take longer depending on system performance.
- If package installation fails, check the Python version and install compatible library versions from `requirements.txt`.

## Author

**258744V**

Capstone project on predictive maintenance and Remaining Useful Life prediction for turbofan engines.
