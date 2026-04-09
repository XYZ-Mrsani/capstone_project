import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDAAnalyzer:
    
    def __init__(self, output_dir='reports/eda'):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_data_overview(self, df_train, df_test):

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        train_sizes = df_train.groupby('Unit').size()
        axes[0, 0].hist(train_sizes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Cycles per Engine')
        axes[0, 0].set_ylabel('Number of Engines')
        axes[0, 0].set_title('Training Data Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        test_sizes = df_test.groupby('Unit').size()
        axes[0, 1].hist(test_sizes, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Cycles per Engine')
        axes[0, 1].set_ylabel('Number of Engines')
        axes[0, 1].set_title('Test Data Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        stats_text = f"""
        Training Set:
        • Engines: {df_train['Unit'].nunique()}
        • Total Cycles: {len(df_train)}
        • Avg Cycles/Engine: {len(df_train) / df_train['Unit'].nunique():.1f}
        • Features: {len([c for c in df_train.columns if c.startswith('Sensor')])}
        
        Test Set:
        • Engines: {df_test['Unit'].nunique()}
        • Total Cycles: {len(df_test)}
        • Avg Cycles/Engine: {len(df_test) / df_test['Unit'].nunique():.1f}
        """
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].axis('off')
        
        setting_cols = ['Setting1', 'Setting2', 'Setting3']
        settings_data = df_train[setting_cols].describe().T
        axes[1, 1].axis('off')
        table_data = settings_data[['mean', 'std', 'min', 'max']].values
        table = axes[1, 1].table(cellText=np.round(table_data, 3),
                                rowLabels=setting_cols,
                                colLabels=['Mean', 'Std', 'Min', 'Max'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        axes[1, 1].set_title('Operational Settings Statistics', pad=20)
        
        plt.tight_layout()
        filepath = self.output_dir / 'data_overview.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_sensor_distributions(self, df, title_suffix='Training'):
        
        sensor_cols = [col for col in df.columns if col.startswith('Sensor') and '_' not in col]
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        fig.suptitle(f'Sensor Value Distributions ({title_suffix})', fontsize=16, fontweight='bold')
        
        for idx, sensor in enumerate(sensor_cols[:12]):
            ax = axes[idx // 4, idx % 4]
            ax.hist(df[sensor].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Sensor Value')
            ax.set_ylabel('Frequency')
            ax.set_title(sensor)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / f'sensor_distributions_{title_suffix.lower()}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_degradation_patterns(self, df, sample_engines=5):

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('Engine Degradation Patterns (Sample Engines)', fontsize=16, fontweight='bold')
        
        engine_ids = df['Unit'].unique()[:sample_engines]
        sensor_cols = [col for col in df.columns if col.startswith('Sensor') and '_' not in col]
        
        for idx, engine_id in enumerate(engine_ids):
            ax = axes[idx // 3, idx % 3]
            engine_data = df[df['Unit'] == engine_id]
            
            mean_sensors = engine_data[sensor_cols].mean(axis=1)
            ax.plot(engine_data['Cycle'], mean_sensors, linewidth=2, color='steelblue', marker='o', markersize=3)
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Mean Sensor Value')
            ax.set_title(f'Engine {engine_id}')
            ax.grid(alpha=0.3)

        axes[1, 2].axis('off')
        
        plt.tight_layout()
        filepath = self.output_dir / 'degradation_patterns.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_sensor_correlations(self, df):

        sensor_cols = [col for col in df.columns if col.startswith('Sensor') and '_' not in col]
        
        corr_matrix = df[sensor_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Sensor Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filepath = self.output_dir / 'sensor_correlations.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_operational_settings(self, df):

        setting_cols = ['Setting1', 'Setting2', 'Setting3']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Operational Settings Distribution', fontsize=14, fontweight='bold')
        
        for idx, setting in enumerate(setting_cols):
            ax = axes[idx]
            ax.hist(df[setting].unique(), bins=20, color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel(f'{setting} Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{setting}')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'operational_settings.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def generate_statistical_summary(self, df_train, df_test):

        sensor_cols = [col for col in df_train.columns if col.startswith('Sensor') and '_' not in col]
        
        summary = {
            'Training Data': {
                'Total Engines': df_train['Unit'].nunique(),
                'Total Cycles': len(df_train),
                'Avg Cycles per Engine': f"{len(df_train) / df_train['Unit'].nunique():.1f}",
                'Min Cycles per Engine': df_train.groupby('Unit').size().min(),
                'Max Cycles per Engine': df_train.groupby('Unit').size().max(),
            },
            'Test Data': {
                'Total Engines': df_test['Unit'].nunique(),
                'Total Cycles': len(df_test),
                'Avg Cycles per Engine': f"{len(df_test) / df_test['Unit'].nunique():.1f}",
                'Min Cycles per Engine': df_test.groupby('Unit').size().min(),
                'Max Cycles per Engine': df_test.groupby('Unit').size().max(),
            },
            'Sensor Statistics': {
                'Total Sensors': len(sensor_cols),
                'Mean Sensor Value': f"{df_train[sensor_cols].mean().mean():.3f}",
                'Std Sensor Value': f"{df_train[sensor_cols].std().mean():.3f}",
                'Min Sensor Value': f"{df_train[sensor_cols].min().min():.3f}",
                'Max Sensor Value': f"{df_train[sensor_cols].max().max():.3f}",
            }
        }
        
        return summary
    
    def run_full_eda(self, df_train, df_test):
        
        logger.info("Starting EDA pipeline...")
        
        self.plot_data_overview(df_train, df_test)
        self.plot_sensor_distributions(df_train, 'Training')
        self.plot_sensor_distributions(df_test, 'Test')
        self.plot_degradation_patterns(df_train)
        self.plot_sensor_correlations(df_train)
        self.plot_operational_settings(df_train)
        
        summary = self.generate_statistical_summary(df_train, df_test)
        
        logger.info("EDA pipeline complete")
        return summary
