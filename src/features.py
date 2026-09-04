"""
Feature engineering script for the AQI prediction project.
Reads cleaned data, engineers time and lag features, checks multicollinearity,
encodes categorical variables, and saves the final feature dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

def get_season(month: int) -> str:
    """
    Determine the season based on the month.
    Winter (Dec=12, Jan=1, Feb=2)
    Summer (Mar-May=3,4,5)
    Monsoon (Jun-Sep=6,7,8,9)
    Post-Monsoon (Oct-Nov=10,11)
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Post-Monsoon'
    return 'Unknown'



if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    input_file = data_dir / 'cleaned_aqi.csv'
    output_file = data_dir / 'features.csv'
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Parse Date as datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Sort by City then Date
    df.sort_values(by=['City', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 3. Extract time features
    print("Extracting time features...")
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['season'] = df['month'].apply(get_season)
    
    # 4. Create lag features per city
    print("Creating lag features...")
    df['AQI_lag_1'] = df.groupby('City')['AQI'].shift(1)
    df['AQI_lag_7'] = df.groupby('City')['AQI'].shift(7)
    df['PM25_lag_1'] = df.groupby('City')['PM2.5'].shift(1)
    
    print("Dropping rows with NaN lag features...")
    df.dropna(subset=['AQI_lag_1', 'AQI_lag_7', 'PM25_lag_1'], inplace=True)
    
    # 5. Multicollinearity check
    print("Performing multicollinearity check (VIF)...")
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    
    while True:
        # Use only pollutants that are still in the dataframe
        current_pollutants = [p for p in pollutants if p in df.columns]
        
        # Calculate VIF on rows without NaNs in the current pollutant columns
        temp_df = df[current_pollutants].dropna()
        if temp_df.empty:
            break
            
        vif_data = []
        for i, col in enumerate(current_pollutants):
            vif = variance_inflation_factor(temp_df.values, i)
            vif_data.append((col, vif))
            
        # Find feature with maximum VIF
        max_vif_col, max_vif_val = max(vif_data, key=lambda x: x[1])
        
        if max_vif_val > 10:
            print(f"Dropping '{max_vif_col}' (VIF = {max_vif_val:.2f} > 10)")
            df.drop(columns=[max_vif_col], inplace=True)
        else:
            print("No more pollutants with VIF > 10.")
            print("Remaining pollutant VIFs:")
            for col, vif in vif_data:
                print(f"  {col}: {vif:.2f}")
            break
            
    # 6. Encode City and Season
    print("One-hot encoding City and season...")
    df = pd.get_dummies(df, columns=['City', 'season'], drop_first=True)
    
    # 7. Drop non-feature columns
    print("Dropping non-feature columns...")
    cols_to_drop = ['Date', 'AQI_Bucket']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # 8. Save to features.csv
    print(f"Saving engineered features to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # 9. Print final shape and columns
    print(f"\nFinal feature DataFrame shape: {df.shape}")
    print("Columns included:")
    for col in df.columns:
        print(f" - {col}")
