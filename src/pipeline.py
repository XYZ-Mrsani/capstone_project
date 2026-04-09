import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import CMAPSSDataLoader, DataPreprocessor, FeatureEngineer, prepare_training_data, get_feature_columns
from eda import EDAAnalyzer
from baseline_models import BaselineModels
from advanced_models import AdvancedModels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RULPredictionPipeline:
    
    def __init__(self, data_dir='data', output_dir='outputs'):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        if self.output_dir.exists():
            import shutil
            logger.info(f"Cleaning previous outputs in {self.output_dir}")
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logger.warning(f"Failed to clear output directory: {e}")
                
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = CMAPSSDataLoader(str(self.data_dir))
        self.preprocessor = DataPreprocessor()
        self.eda_analyzer = EDAAnalyzer(str(self.output_dir / 'eda'))
        self.baseline_models = BaselineModels(str(self.output_dir / 'models'))
        self.advanced_models = AdvancedModels(str(self.output_dir / 'models'))
        
        self.df_train = None
        self.df_test_full = None
        self.df_test = None
        self.rul_labels = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self, dataset='FD001'):
        logger.info(f"Loading {dataset} dataset...")
        
        self.df_train = self.data_loader.load_train_data(dataset)
        self.df_test = self.data_loader.load_test_data(dataset)
        self.rul_labels = self.data_loader.load_rul_labels(dataset)
        
        logger.info("Data loading complete")
    
    def preprocess_data(self):
        logger.info("Preprocessing data...")
        
        self.df_train, self.df_test = self.preprocessor.preprocess(self.df_train, self.df_test)
        
        logger.info("Preprocessing complete")
    
    def feature_engineering(self):
        logger.info("Creating engineered features...")
        
        self.df_train = FeatureEngineer.create_features(self.df_train)
        self.df_test = FeatureEngineer.create_features(self.df_test)
        
        logger.info("Feature engineering complete")
    
    def prepare_data_for_modeling(self):
        logger.info("Preparing data for modeling...")
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.df_test_full = prepare_training_data(
            self.df_train, self.df_test, self.rul_labels
        )
        
        logger.info("Data preparation complete")
    
    def run_eda(self):
        logger.info("Running EDA...")
        
        summary = self.eda_analyzer.run_full_eda(self.df_train, self.df_test)
        
        logger.info("EDA complete")
        return summary
    
    def train_baseline_models(self):
        logger.info("Training baseline models...")
        
        baseline_results = self.baseline_models.train_and_evaluate(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        logger.info("Baseline models training complete")
        return baseline_results
    
    def train_advanced_models(self):
        logger.info("Training advanced models...")
        
        n_train = len(self.X_train)
        n_val = int(0.2 * n_train)
        
        X_train_split = self.X_train.iloc[:-n_val]
        y_train_split = self.y_train[:-n_val]
        X_val = self.X_train.iloc[-n_val:]
        y_val = self.y_train[-n_val:]
        
        advanced_results = self.advanced_models.train_and_evaluate(
            X_train_split, y_train_split, self.X_test, self.y_test,
            self.df_test_full, X_val, y_val
        )
        
        logger.info("Advanced models training complete")
        return advanced_results
    
    def generate_comparison_report(self):
        logger.info("Generating model comparison report...")
        
        baseline_summary = self.baseline_models.get_results_summary()
        
        advanced_summary = self.advanced_models.get_results_summary()
        
        comparison = pd.concat([baseline_summary, advanced_summary], ignore_index=True)
        
        report_path = self.output_dir / 'model_comparison.csv'
        try:
            comparison.to_csv(report_path, index=False)
            logger.info(f"Saved model comparison to {report_path}")
        except PermissionError:
            import time
            fallback_path = self.output_dir / f'model_comparison_{int(time.time())}.csv'
            comparison.to_csv(fallback_path, index=False)
            logger.warning(f"Permission denied for {report_path} (file might be open). Saved to {fallback_path} instead.")
        
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info("\n" + comparison.to_string(index=False))
        logger.info("=" * 80)
        
        return comparison
    
    def generate_feature_importance_plot(self):
        logger.info("Generating feature importance plot...")
        
        feature_importance_df = self.baseline_models.results.get('Feature Importance')
        
        if feature_importance_df is not None:
            top_features = feature_importance_df.head(20)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(top_features)), top_features['Importance %'].values, color='steelblue')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'].values)
            ax.set_xlabel('Importance (%)')
            ax.set_title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(alpha=0.3, axis='x')
            
            plt.tight_layout()
            plot_path = self.output_dir / 'eda' / 'feature_importance.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {plot_path}")
            plt.close()
    
    def run_full_pipeline(self, dataset='FD001'):

        logger.info("=" * 80)
        logger.info("STARTING RUL PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        self.load_data(dataset)
        
        self.preprocess_data()
        
        self.feature_engineering()
        
        self.prepare_data_for_modeling()
        
        self.run_eda()
        
        self.train_baseline_models()
        
        self.train_advanced_models()
        
        comparison = self.generate_comparison_report()
        self.generate_feature_importance_plot()
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return comparison


if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent
    pipeline = RULPredictionPipeline(data_dir=root_dir / 'data', output_dir=root_dir / 'outputs')
    results = pipeline.run_full_pipeline(dataset='FD001')
