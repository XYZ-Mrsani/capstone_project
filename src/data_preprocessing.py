import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CMAPSSDataLoader:
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self):

        setting_names = ['Setting1', 'Setting2', 'Setting3']
        sensor_names = [f'Sensor{i}' for i in range(1, 22)]
        return ['Unit', 'Cycle'] + setting_names + sensor_names
    
    def load_train_data(self, dataset='FD001'):

        filepath = os.path.join(self.data_dir, f'train_{dataset}.txt')
        logger.info(f"Loading training data from {filepath}")
        
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=self.feature_names)
        logger.info(f"Loaded {len(df)} training samples from {df['Unit'].nunique()} engines")
        return df
    
    def load_test_data(self, dataset='FD001'):

        filepath = os.path.join(self.data_dir, f'test_{dataset}.txt')
        logger.info(f"Loading test data from {filepath}")
        
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=self.feature_names)
        logger.info(f"Loaded {len(df)} test samples from {df['Unit'].nunique()} engines")
        return df
    
    def load_rul_labels(self, dataset='FD001'):
        filepath = os.path.join(self.data_dir, f'RUL_{dataset}.txt')
        logger.info(f"Loading RUL labels from {filepath}")
        
        rul = pd.read_csv(filepath, sep=r'\s+', header=None, names=['RUL'])
        logger.info(f"Loaded {len(rul)} RUL labels")
        return rul['RUL'].values


class DataPreprocessor:
    
    def __init__(self):
        self.scaler_settings = StandardScaler()
        self.scaler_sensors = StandardScaler()
        self.constant_sensors = []
        
    def identify_constant_sensors(self, df, threshold=0.01):

        sensor_cols = [col for col in df.columns if col.startswith('Sensor')]
        constant_sensors = []
        
        for sensor in sensor_cols:
            variance = df[sensor].var()
            if variance < threshold:
                constant_sensors.append(sensor)
                logger.info(f"Identified constant sensor: {sensor} (variance: {variance:.6f})")
        
        self.constant_sensors = constant_sensors
        return constant_sensors
    
    def remove_constant_sensors(self, df):
        df_cleaned = df.drop(columns=self.constant_sensors)
        logger.info(f"Removed {len(self.constant_sensors)} constant sensors")
        return df_cleaned
    
    def normalize_data(self, df_train, df_test=None, fit=True):

        id_cols = ['Unit', 'Cycle']
        setting_cols = ['Setting1', 'Setting2', 'Setting3']
        sensor_cols = [col for col in df_train.columns if col.startswith('Sensor')]
        
        if fit:
            self.scaler_settings.fit(df_train[setting_cols])
        
        df_train[setting_cols] = self.scaler_settings.transform(df_train[setting_cols])
        if df_test is not None:
            df_test[setting_cols] = self.scaler_settings.transform(df_test[setting_cols])
        
        if fit:
            self.scaler_sensors.fit(df_train[sensor_cols])
        
        df_train[sensor_cols] = self.scaler_sensors.transform(df_train[sensor_cols])
        if df_test is not None:
            df_test[sensor_cols] = self.scaler_sensors.transform(df_test[sensor_cols])
        
        logger.info("Normalized operational settings and sensor readings")
        return df_train, df_test
    
    def preprocess(self, df_train, df_test=None):

        logger.info("Starting preprocessing pipeline...")
        
        self.identify_constant_sensors(df_train)
        df_train = self.remove_constant_sensors(df_train)
        if df_test is not None:
            df_test = self.remove_constant_sensors(df_test)
        
        df_train, df_test = self.normalize_data(df_train, df_test, fit=True)
        
        logger.info("Preprocessing complete")
        return df_train, df_test


class FeatureEngineer:
    
    @staticmethod
    def add_rolling_window_features(df, windows=[5, 10, 15]):
        df = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('Sensor')]
        
        for window in windows:
            for sensor in sensor_cols:
                df[f'{sensor}_roll_mean_{window}'] = df.groupby('Unit')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{sensor}_roll_std_{window}'] = df.groupby('Unit')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                )
        
        logger.info(f"Added rolling window features for windows: {windows}")
        return df
    
    @staticmethod
    def add_degradation_features(df):

        df = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('Sensor')]

        max_cycles = df.groupby('Unit')['Cycle'].transform('max')
        df['Time_Since_Start'] = df['Cycle'] / max_cycles

        logger.info("Added degradation features")
        return df
    
    @staticmethod
    def add_sensor_fusion_features(df):

        df = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('Sensor') and '_' not in col]
        
        df['Sensor_Mean'] = df[sensor_cols].mean(axis=1)
        df['Sensor_Std'] = df[sensor_cols].std(axis=1)
        df['Sensor_Max'] = df[sensor_cols].max(axis=1)
        df['Sensor_Min'] = df[sensor_cols].min(axis=1)
        
        logger.info("Added sensor fusion features")
        return df
    
    @staticmethod
    def create_features(df):

        logger.info("Creating engineered features...")
        df = FeatureEngineer.add_rolling_window_features(df, windows=[5, 10, 15])
        df = FeatureEngineer.add_degradation_features(df)
        df = FeatureEngineer.add_sensor_fusion_features(df)
        logger.info("Feature engineering complete")
        return df


def prepare_training_data(df_train, df_test, rul_labels):

    logger.info("Preparing training and test data.")

    train_parts = []

    for engine_id in sorted(df_train['Unit'].unique()):
        engine_data = df_train[df_train['Unit'] == engine_id].copy()
        max_cycle = engine_data['Cycle'].max()

        engine_data['RUL'] = max_cycle - engine_data['Cycle']
        train_parts.append(engine_data)

    X_train = pd.concat(train_parts, ignore_index=True)
    y_train = X_train['RUL'].values

    test_last_parts = []
    y_test_last = []

    test_full_parts = []

    for idx, engine_id in enumerate(sorted(df_test['Unit'].unique())):
        engine_data = df_test[df_test['Unit'] == engine_id].copy()

        max_cycle_observed = engine_data['Cycle'].max()
        final_rul = rul_labels[idx]

        engine_data['RUL'] = (max_cycle_observed - engine_data['Cycle']) + final_rul
        test_full_parts.append(engine_data)

        last_cycle_data = engine_data.iloc[[-1]].copy()
        test_last_parts.append(last_cycle_data)
        y_test_last.append(final_rul)

    X_test_last = pd.concat(test_last_parts, ignore_index=True)
    y_test_last = np.array(y_test_last)
    df_test_full_labeled = pd.concat(test_full_parts, ignore_index=True)

    logger.info(f"Training set: {len(X_train)} samples, {X_train['Unit'].nunique()} engines")
    logger.info(f"Tabular test set: {len(X_test_last)} samples, {X_test_last['Unit'].nunique()} engines")
    logger.info(f"Sequence test set: {len(df_test_full_labeled)} samples, {df_test_full_labeled['Unit'].nunique()} engines")

    return X_train, y_train, X_test_last, y_test_last, df_test_full_labeled


def get_feature_columns(df):

    exclude_cols = ['Unit', 'Cycle', 'RUL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols
