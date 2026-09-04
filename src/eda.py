"""
Exploratory Data Analysis script for AQI dataset.
Generates various plots to visualize trends, correlations, and distributions,
and saves them to the reports/figures directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
try:
    from src.features import get_season
except ImportError:
    from features import get_season



def load_data(filepath):
    """Loads the cleaned AQI dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['Date'])
    return df

def plot_aqi_trends(df, output_dir):
    """Plots monthly average AQI trend per city."""
    print("Generating AQI trends plot...")
    plt.figure(figsize=(12, 8))
    
    # Extract year and month for grouping
    df_monthly = df.set_index('Date').groupby(['City', pd.Grouper(freq='M')])['AQI'].mean().reset_index()
    
    sns.lineplot(data=df_monthly, x='Date', y='AQI', hue='City')
    
    plt.title('Monthly Average AQI Trends by City')
    plt.xlabel('Date')
    plt.ylabel('Average AQI')
    plt.tight_layout()
    plt.savefig(output_dir / 'aqi_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved aqi_trends.png")

def plot_correlation_heatmap(df, output_dir):
    """Plots a correlation heatmap for numeric pollutant columns and AQI."""
    print("Generating correlation heatmap...")
    plt.figure(figsize=(12, 8))
    
    # Select numeric columns relevant for correlation
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
    cols_to_use = [col for col in pollutants if col in df.columns]
    
    corr = df[cols_to_use].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Heatmap of Pollutants and AQI')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved correlation_heatmap.png")



def plot_seasonal_patterns(df, output_dir):
    """Plots AQI distributions by season."""
    print("Generating seasonal patterns plot...")
    plt.figure(figsize=(12, 8))
    
    # Add season column
    df['Season'] = df['Date'].dt.month.apply(get_season)
    season_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    
    sns.boxplot(data=df, x='Season', y='AQI', order=season_order)
    plt.title('AQI Distribution by Season')
    plt.xlabel('Season')
    plt.ylabel('AQI')
    plt.tight_layout()
    plt.savefig(output_dir / 'seasonal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved seasonal_patterns.png")

def plot_pm25_vs_aqi(df, output_dir):
    """Plots scatter plot of PM2.5 vs AQI colored by city."""
    print("Generating PM2.5 vs AQI scatter plot...")
    plt.figure(figsize=(12, 8))
    
    if 'PM2.5' in df.columns and 'AQI' in df.columns:
        sns.scatterplot(data=df, x='PM2.5', y='AQI', hue='City', alpha=0.6)
        plt.title('PM2.5 vs AQI by City')
        plt.xlabel('PM2.5')
        plt.ylabel('AQI')
    else:
        print("PM2.5 or AQI column not found. Skipping plot.")
        
    plt.tight_layout()
    plt.savefig(output_dir / 'pm25_vs_aqi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pm25_vs_aqi.png")

def plot_missing_data(df, output_dir):
    """Plots a bar chart showing the count of zero-values per column."""
    print("Generating missing data (zero values) plot...")
    plt.figure(figsize=(12, 8))
    
    # Calculate number of zero values for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    zero_counts = (numeric_df == 0).sum()
    
    # Plot
    zero_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Count of Zero Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Count of Zeros')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved missing_data.png")

def main():
    """Main function to run the EDA process."""
    data_path = Path('data/cleaned_aqi.csv')
    
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found. Please ensure you are running from the project root.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    
    # Generate all plots
    plot_aqi_trends(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_seasonal_patterns(df, output_dir)
    plot_pm25_vs_aqi(df, output_dir)
    plot_missing_data(df, output_dir)
    
    print(f"All EDA plots generated and saved to {output_dir}")

if __name__ == '__main__':
    main()
