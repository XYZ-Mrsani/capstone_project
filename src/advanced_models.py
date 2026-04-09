import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedModels:
    
    def __init__(self, output_dir='models'):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.xgb_model = None
        self.lstm_model = None
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
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):

        logger.info("Training XGBoost model...")
        
        X_features, y = self.prepare_features(X_train, y_train)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        dtrain = xgb.DMatrix(X_features, label=y)
        
        dval = None
        evals = None
        if X_val is not None and y_val is not None:
            X_val_features, y_val_clean = self.prepare_features(X_val, y_val)
            dval = xgb.DMatrix(X_val_features, label=y_val_clean)
            evals = [(dtrain, 'train'), (dval, 'eval')]

        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=20 if evals else None,
            verbose_eval=False
        )
        
        model_path = self.output_dir / 'xgboost.pkl'
        joblib.dump(self.xgb_model, model_path)
        logger.info(f"Saved XGBoost model to {model_path}")
        
        return self.xgb_model
    
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None, sequence_length=10):

        logger.info("Training LSTM model.")

        train_df = X_train.copy()
        train_df['RUL'] = y_train

        feature_cols = [col for col in train_df.columns if col not in ['Unit', 'Cycle', 'RUL']]

        X_train_seq, y_train_seq = self._create_sequences_by_engine(
            train_df, sequence_length, feature_cols
        )
        logger.info(f"Created {len(X_train_seq)} training sequences")

        if len(X_train_seq) == 0:
            logger.warning("No training sequences were created for LSTM.")
            return None

        model = Sequential([
            layers.Input(shape=(sequence_length, len(feature_cols))),
            layers.LSTM(64, activation='tanh', return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        validation_data = None
        monitor_metric = 'loss'

        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df['RUL'] = y_val

            X_val_seq, y_val_seq = self._create_sequences_by_engine(
                val_df, sequence_length, feature_cols
            )

            if len(X_val_seq) > 0:
                validation_data = (X_val_seq, y_val_seq)
                monitor_metric = 'val_loss'
                logger.info(f"Created {len(X_val_seq)} validation sequences")

        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=32,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=0
        )

        self.lstm_model = model

        model_path = self.output_dir / 'lstm_model.h5'
        model.save(model_path)
        logger.info(f"Saved LSTM model to {model_path}")

        return model
    
    @staticmethod
    def _create_sequences(X, y, sequence_length):

        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
        
    @staticmethod
    def _create_sequences_by_engine(df, sequence_length, feature_cols):

        X_seq = []
        y_seq = []

        for engine_id in sorted(df['Unit'].unique()):
            engine_df = df[df['Unit'] == engine_id].sort_values('Cycle').copy()

            X_engine = engine_df[feature_cols].fillna(engine_df[feature_cols].mean()).values
            y_engine = engine_df['RUL'].values

            if len(engine_df) < sequence_length:
                continue

            for i in range(len(engine_df) - sequence_length + 1):
                X_seq.append(X_engine[i:i + sequence_length])
                y_seq.append(y_engine[i + sequence_length - 1])

        return np.array(X_seq), np.array(y_seq)
    
    def evaluate_xgboost(self, X_test, y_test):

        if self.xgb_model is None:
            logger.warning("XGBoost model not trained yet")
            return None
        
        X_features = self.prepare_features(X_test)
        dtest = xgb.DMatrix(X_features)
        
        y_pred = self.xgb_model.predict(dtest)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        nasa_score = self._calculate_nasa_score(y_test, y_pred)
        
        metrics = {
            'Model': 'XGBoost',
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'NASA Score': nasa_score,
            'Predictions': y_pred
        }
        
        logger.info("XGBoost Evaluation:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  NASA Score: {nasa_score:.4f}")
        
        return metrics
    
    def evaluate_lstm(self, df_test_full, sequence_length=10):

        if self.lstm_model is None:
            logger.warning("LSTM model not trained yet")
            return None

        feature_cols = [col for col in df_test_full.columns if col not in ['Unit', 'Cycle', 'RUL']]

        X_test_seq, y_test_seq = self._create_sequences_by_engine(
            df_test_full, sequence_length, feature_cols
        )

        if len(X_test_seq) == 0:
            logger.warning("Not enough test data to create LSTM sequences")
            return None

        y_pred = self.lstm_model.predict(X_test_seq, verbose=0).flatten()

        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
        mae = mean_absolute_error(y_test_seq, y_pred)
        r2 = r2_score(y_test_seq, y_pred)
        nasa_score = self._calculate_nasa_score(y_test_seq, y_pred)

        metrics = {
            'Model': 'LSTM',
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'NASA Score': nasa_score,
            'Predictions': y_pred
        }

        logger.info("LSTM Evaluation:")
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
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, df_test_full, X_val=None, y_val=None):

        logger.info("=" * 60)
        logger.info("ADVANCED MODELS TRAINING AND EVALUATION")
        logger.info("=" * 60)

        self.train_xgboost(X_train, y_train, X_val, y_val)
        xgb_results = self.evaluate_xgboost(X_test, y_test)
        self.results['XGBoost'] = xgb_results
        
        logger.info("-" * 60)
        
        self.train_lstm(X_train, y_train, X_val, y_val)
        lstm_results = self.evaluate_lstm(df_test_full)
        self.results['LSTM'] = lstm_results
        
        return self.results
    
    def get_results_summary(self):

        summary_data = []
        
        for model_name, results in self.results.items():
            if results is not None:
                summary_data.append({
                    'Model': results['Model'],
                    'RMSE': f"{results['RMSE']:.4f}",
                    'MAE': f"{results['MAE']:.4f}",
                    'R² Score': f"{results['R² Score']:.4f}",
                    'NASA Score': f"{results['NASA Score']:.4f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
