import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import CMAPSSDataLoader, DataPreprocessor, FeatureEngineer, get_feature_columns

st.set_page_config(
    page_title="RUL Prediction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RULDashboard:
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / 'data'
        self.models_dir = self.root_dir / 'outputs' / 'models'
        self.eda_dir = self.root_dir / 'outputs' / 'eda'
        
        self.models = self._load_models()
        
        self.data_loader = CMAPSSDataLoader(str(self.data_dir))
        self.df_train = self.data_loader.load_train_data('FD001')
        self.df_test = self.data_loader.load_test_data('FD001')
        self.rul_labels = self.data_loader.load_rul_labels('FD001')
    
    def _load_models(self):
        models = {}
        
        try:
            if (self.models_dir / 'linear_regression.pkl').exists():
                models['Linear Regression'] = joblib.load(self.models_dir / 'linear_regression.pkl')
        except Exception as e:
            st.warning(f"Could not load Linear Regression model: {e}")
        
        try:
            if (self.models_dir / 'random_forest.pkl').exists():
                models['Random Forest'] = joblib.load(self.models_dir / 'random_forest.pkl')
        except Exception as e:
            st.warning(f"Could not load Random Forest model: {e}")
        
        try:
            if (self.models_dir / 'xgboost.pkl').exists():
                models['XGBoost'] = joblib.load(self.models_dir / 'xgboost.pkl')
        except Exception as e:
            st.warning(f"Could not load XGBoost model: {e}")
        
        return models
    
    def prepare_engine_data(self, engine_id):
        engine_data = self.df_test[self.df_test['Unit'] == engine_id].copy()
        
        if len(engine_data) == 0:
            return None
        
        last_cycle_data = engine_data.iloc[-1:].copy()
        
        preprocessor = DataPreprocessor()
        preprocessor.identify_constant_sensors(self.df_train)
        last_cycle_data = preprocessor.remove_constant_sensors(last_cycle_data)
        
        last_cycle_data = FeatureEngineer.create_features(last_cycle_data)
        
        return last_cycle_data, engine_data
    
    def predict_rul(self, engine_data, model_name):
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        exclude_cols = ['Unit', 'Cycle', 'RUL']
        feature_cols = [col for col in engine_data.columns if col not in exclude_cols]
        X = engine_data[feature_cols].fillna(engine_data[feature_cols].mean())
        
        try:
            if model_name == 'XGBoost':
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                prediction = model.predict(dtest)[0]
            else:
                prediction = model.predict(X)[0]
            
            return max(0, prediction)  
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def plot_engine_health(self, engine_data_full):
        sensor_cols = [col for col in engine_data_full.columns if col.startswith('Sensor') and '_' not in col]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        mean_sensors = engine_data_full[sensor_cols].mean(axis=1)
        axes[0].plot(engine_data_full['Cycle'], mean_sensors, linewidth=2, color='steelblue', marker='o', markersize=4)
        axes[0].set_xlabel('Cycle')
        axes[0].set_ylabel('Mean Sensor Value')
        axes[0].set_title('Engine Health Trend (Mean Sensor Reading)')
        axes[0].grid(alpha=0.3)
        
        last_cycle_sensors = engine_data_full[sensor_cols].iloc[-1]
        axes[1].bar(range(len(last_cycle_sensors)), last_cycle_sensors.values, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Sensor ID')
        axes[1].set_ylabel('Sensor Reading')
        axes[1].set_title('Sensor Readings at Last Cycle')
        axes[1].set_xticks(range(len(last_cycle_sensors)))
        axes[1].set_xticklabels([f'S{i+1}' for i in range(len(last_cycle_sensors))], rotation=45)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self):
        comparison_file = self.root_dir / 'outputs' / 'model_comparison.csv'
        
        if not comparison_file.exists():
            st.warning("Model comparison file not found")
            return None
        
        df_comparison = pd.read_csv(comparison_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RMSE comparison
        axes[0, 0].bar(df_comparison['Model'], df_comparison['RMSE'].astype(float), color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Model Comparison: RMSE (Lower is Better)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # MAE comparison
        axes[0, 1].bar(df_comparison['Model'], df_comparison['MAE'].astype(float), color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Model Comparison: MAE (Lower is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # R² Score comparison
        axes[1, 0].bar(df_comparison['Model'], df_comparison['R² Score'].astype(float), color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Comparison: R² Score (Higher is Better)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # NASA Score comparison
        axes[1, 1].bar(df_comparison['Model'], df_comparison['NASA Score'].astype(float), color='plum', edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('NASA Score')
        axes[1, 1].set_title('Model Comparison: NASA Score (Lower is Better)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def run(self):
        st.title("✈️ Turbofan Engine RUL Prediction Dashboard")
        st.markdown("---")
        
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Page", ["Home", "Engine Analysis", "Model Comparison", "EDA Visualizations"])
        
        if page == "Home":
            self._page_home()
        elif page == "Engine Analysis":
            self._page_engine_analysis()
        elif page == "Model Comparison":
            self._page_model_comparison()
        elif page == "EDA Visualizations":
            self._page_eda_visualizations()
    
    def _page_home(self):
        st.header("Welcome to the RUL Prediction System")
        
        st.markdown("""
        This dashboard provides an interactive interface for predicting the **Remaining Useful Life (RUL)** 
        of aircraft turbofan engines using machine learning models trained on the NASA C-MAPSS dataset.
        
        ### Key Features:
        - **Engine Analysis**: Predict RUL for individual engines using different models
        - **Model Comparison**: Compare performance metrics across all trained models
        - **EDA Visualizations**: Explore data patterns and relationships
        
        ### System Overview:
        The system consists of:
        1. **Data Preprocessing**: Cleaning, normalization, and feature engineering
        2. **Baseline Models**: Linear Regression and Random Forest
        3. **Advanced Models**: XGBoost and LSTM neural networks
        4. **Evaluation Metrics**: RMSE, MAE, R² Score, and NASA Score
        
        ### How to Use:
        1. Navigate to "Engine Analysis" to predict RUL for a specific engine
        2. Use "Model Comparison" to see which model performs best
        3. Check "EDA Visualizations" to understand the data
        """)
        
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Engines", self.df_train['Unit'].nunique())
        with col2:
            st.metric("Test Engines", self.df_test['Unit'].nunique())
        with col3:
            st.metric("Total Features", len([c for c in self.df_train.columns if c.startswith('Sensor')]))
        with col4:
            st.metric("Operational Settings", 3)
    
    def _page_engine_analysis(self):
        st.header("Engine RUL Analysis")
        
        engine_id = st.sidebar.slider("Select Engine ID", 1, self.df_test['Unit'].max(), 1)
        
        result = self.prepare_engine_data(engine_id)
        
        if result is None:
            st.error(f"No data found for Engine {engine_id}")
            return
        
        engine_data, engine_data_full = result
        
        st.subheader(f"Engine {engine_id} Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cycles", len(engine_data_full))
        with col2:
            st.metric("Last Cycle", engine_data_full['Cycle'].max())
        with col3:
            st.metric("Actual RUL (Test Set)", self.rul_labels[engine_id - 1])
        
        st.subheader("RUL Predictions by Model")
        
        predictions = {}
        for model_name in self.models.keys():
            rul_pred = self.predict_rul(engine_data, model_name)
            if rul_pred is not None:
                predictions[model_name] = rul_pred
        
        if predictions:
            cols = st.columns(len(predictions))
            for idx, (model_name, rul_pred) in enumerate(predictions.items()):
                with cols[idx]:
                    st.metric(model_name, f"{rul_pred:.1f} cycles")
        
        st.subheader("Engine Health Trends")
        fig = self.plot_engine_health(engine_data_full)
        st.pyplot(fig)
        
        with st.expander("View Sensor Data"):
            sensor_cols = [col for col in engine_data_full.columns if col.startswith('Sensor')]
            st.dataframe(engine_data_full[['Unit', 'Cycle'] + sensor_cols].tail(10))
    
    def _page_model_comparison(self):
        st.header("Model Performance Comparison")
        
        comparison_file = self.root_dir / 'outputs' / 'model_comparison.csv'
        
        if not comparison_file.exists():
            st.error("Model comparison file not found. Please run the pipeline first.")
            return
        
        df_comparison = pd.read_csv(comparison_file)
        
        st.subheader("Performance Metrics")
        st.dataframe(df_comparison, use_container_width=True)
        
        st.subheader("Visual Comparison")
        fig = self.plot_model_comparison()
        if fig:
            st.pyplot(fig)
        
        st.subheader("Model Insights")
        
        best_rmse_idx = df_comparison['RMSE'].astype(float).idxmin()
        best_mae_idx = df_comparison['MAE'].astype(float).idxmin()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Best RMSE**: {df_comparison.loc[best_rmse_idx, 'Model']} ({df_comparison.loc[best_rmse_idx, 'RMSE']})")
        
        with col2:
            st.info(f"**Best MAE**: {df_comparison.loc[best_mae_idx, 'Model']} ({df_comparison.loc[best_mae_idx, 'MAE']})")
    
    def _page_eda_visualizations(self):
        st.header("Exploratory Data Analysis")
        
        eda_files = [
            ('data_overview.png', 'Dataset Overview'),
            ('sensor_distributions_training.png', 'Sensor Distributions (Training)'),
            ('degradation_patterns.png', 'Engine Degradation Patterns'),
            ('sensor_correlations.png', 'Sensor Correlations'),
            ('operational_settings.png', 'Operational Settings'),
            ('feature_importance.png', 'Feature Importance'),
        ]
        
        for filename, title in eda_files:
            filepath = self.eda_dir / filename
            if filepath.exists():
                st.subheader(title)
                st.image(str(filepath), use_column_width=True)
            else:
                st.warning(f"{title} not found")


def main():
    dashboard = RULDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
