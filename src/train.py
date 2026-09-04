import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

def load_and_preprocess_data(filepath: Path) -> pd.DataFrame:
    """
    Load feature data and preprocess it for modeling.
    
    Args:
        filepath (Path): Path to the features.csv file.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Drop AQI_Bucket if it exists
    if 'AQI_Bucket' in df.columns:
        df = df.drop(columns=['AQI_Bucket'])
    
    # Convert boolean columns (from one-hot encoding) to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        logger.info(f"Converting {len(bool_cols)} boolean columns to int: {bool_cols.tolist()}")
        df[bool_cols] = df[bool_cols].astype(int)
        
    # Encode season column if it exists and is object type
    if 'season' in df.columns and df['season'].dtype == 'object':
        df['season'] = pd.factorize(df['season'])[0]
        logger.info("Encoded 'season' column.")
        
    # Drop any remaining non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.info(f"Dropping remaining non-numeric columns: {non_numeric_cols.tolist()}")
        df = df.drop(columns=non_numeric_cols)
        
    # Handle NaNs that might arise from lag features or missing data
    # Simplest approach is to dropna or fillna. Using dropna for a clean dataset.
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    if initial_len != final_len:
        logger.info(f"Dropped {initial_len - final_len} rows with NaN values.")
        
    return df



def plot_feature_importance(model, feature_names: list, save_path: Path):
    """
    Plot and save feature importance for trained models.
    Works with tree-based models (feature_importances_) and
    linear models (absolute coef_ values).
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        save_path (Path): Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {type(model).__name__} does not have feature_importances_.")
        return
    importances = model.feature_importances_
    title = 'Top 15 Feature Importances'
    
    indices = np.argsort(importances)[::-1]
    
    # Take top 15
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(top_n), top_importances, align='center', color='steelblue')
    plt.xticks(range(top_n), top_features, rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")

def main():
    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        return
        
    # 1 & 2. Load and preprocess
    df = load_and_preprocess_data(features_path)
    
    if 'AQI' not in df.columns:
        logger.error("Target column 'AQI' not found in the dataset.")
        return
        
    X = df.drop(columns=['AQI'])
    y = df['AQI']
    
    # 3. Time-aware split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Time-aware split created.")
    logger.info(f"Train size: {len(X_train)} (80%)")
    logger.info(f"Test size: {len(X_test)} (20%)")
    
    # 4. Train baseline model (LinearRegression)
    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print("-" * 40)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("-" * 40)
    
    best_model = model
    best_model_name = 'Linear Regression'
    best_params = {}
    results = {best_model_name: {'RMSE': rmse, 'MAE': mae, 'R2': r2}}
    final_rmse, final_mae, final_r2 = rmse, mae, r2
    
    # 8. Save outputs
    # Save model
    model_path = MODELS_DIR / "aqi_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save feature columns
    features_path = MODELS_DIR / "feature_columns.pkl"
    feature_cols = X.columns.tolist()
    joblib.dump(feature_cols, features_path)
    logger.info(f"Saved feature columns to {features_path}")
    
    # Save metrics JSON
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_dict = {
        name: {
            'RMSE': float(metrics['RMSE']),
            'MAE': float(metrics['MAE']),
            'R2': float(metrics['R2'])
        }
        for name, metrics in results.items()
    }
    metrics_dict['best_model'] = best_model_name
    metrics_dict['best_params'] = best_params
    metrics_dict['final_tuned_metrics'] = {
        'RMSE': float(final_rmse),
        'MAE': float(final_mae),
        'R2': float(final_r2)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Plot and save feature importance
    plot_path = FIGURES_DIR / "feature_importance.png"
    plot_feature_importance(best_model, feature_cols, plot_path)
    
    logger.info("Training pipeline completed successfully.")

if __name__ == '__main__':
    main()
