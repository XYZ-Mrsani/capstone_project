import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineModels:
    
    def __init__(self, output_dir='models'):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.results = {}
    
    def prepare_features(self, X, y=None):
        
        exclude_cols = ['Unit', 'Cycle', 'RUL']
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        X_features = X[feature_cols].copy()
        
        X_features = X_features.fillna(X_features.mean())
        
        if y is not None:
            return X_features, y
        else:
            return X_features
    
    def train_linear_regression(self, X_train, y_train):

        logger.info("Training Linear Regression model...")
        
        X_features, y = self.prepare_features(X_train, y_train)
        
        self.lr_model.fit(X_features, y)
        
        model_path = self.output_dir / 'linear_regression.pkl'
        joblib.dump(self.lr_model, model_path)
        logger.info(f"Saved Linear Regression model to {model_path}")
        
        return self.lr_model
    
    def train_random_forest(self, X_train, y_train):
        
        logger.info("Training Random Forest model...")
        
        X_features, y = self.prepare_features(X_train, y_train)
        
        self.rf_model.fit(X_features, y)
        
        model_path = self.output_dir / 'random_forest.pkl'
        joblib.dump(self.rf_model, model_path)
        logger.info(f"Saved Random Forest model to {model_path}")
        
        return self.rf_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        
        X_features = self.prepare_features(X_test)
        
        y_pred = model.predict(X_features)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        nasa_score = self._calculate_nasa_score(y_test, y_pred)
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'NASA Score': nasa_score,
            'Predictions': y_pred
        }
        
        logger.info(f"{model_name} Evaluation:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  NASA Score: {nasa_score:.4f}")
        
        return metrics
    
    @staticmethod
    def _calculate_nasa_score(y_true, y_pred):
        
        score = 0
        for true_val, pred_val in zip(y_true, y_pred):
            if pred_val < true_val:
                score += np.exp(-pred_val / 13) - 1
            else:
                score += np.exp(pred_val / 10) - 1
        
        return score
    
    def get_feature_importance(self, model, X_train, feature_names=None):
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        X_features = self.prepare_features(X_train)
        
        if feature_names is None:
            feature_names = X_features.columns
        
        importances = model.feature_importances_
        
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices],
            'Importance %': (importances[indices] / importances.sum()) * 100
        })
        
        logger.info("Top 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance %']:.2f}%")
        
        return importance_df
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        
        logger.info("=" * 60)
        logger.info("BASELINE MODELS TRAINING AND EVALUATION")
        logger.info("=" * 60)
        
        self.train_linear_regression(X_train, y_train)
        lr_results = self.evaluate_model(self.lr_model, X_test, y_test, 'Linear Regression')
        self.results['Linear Regression'] = lr_results
        
        logger.info("-" * 60)
        
        self.train_random_forest(X_train, y_train)
        rf_results = self.evaluate_model(self.rf_model, X_test, y_test, 'Random Forest')
        self.results['Random Forest'] = rf_results
        
        feature_cols = [col for col in X_train.columns if col not in ['Unit', 'Cycle', 'RUL']]
        importance_df = self.get_feature_importance(self.rf_model, X_train, feature_cols)
        self.results['Feature Importance'] = importance_df
        
        logger.info("=" * 60)
        
        return self.results
    
    def get_results_summary(self):
        
        summary_data = []
        
        for model_name, results in self.results.items():
            if model_name != 'Feature Importance':
                summary_data.append({
                    'Model': results['Model'],
                    'RMSE': f"{results['RMSE']:.4f}",
                    'MAE': f"{results['MAE']:.4f}",
                    'R² Score': f"{results['R² Score']:.4f}",
                    'NASA Score': f"{results['NASA Score']:.4f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
