import pandas as pd
import numpy as np
from pathlib import Path

# Calculate project root dynamically based on script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "city_day.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_aqi.csv"

POLLUTANTS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]

# Indian NAQI standards: (low, high, label, color)
AQI_BUCKETS = [
    (0, 50, "Good", "#2ecc71"),
    (51, 100, "Satisfactory", "#a3d977"),
    (101, 200, "Moderate", "#f1c40f"),
    (201, 300, "Poor", "#e67e22"),
    (301, 400, "Very Poor", "#e74c3c"),
    (401, 500, "Severe", "#8b0000"),
]

def get_aqi_bucket(aqi: float) -> tuple:
    """Categorize AQI based on Indian NAQI standards."""
    if pd.isna(aqi) or aqi < 0:
        return ("Invalid", "#999999")
    for low, high, label, color in AQI_BUCKETS:
        if aqi <= high:
            return (label, color)
    return ("Severe", "#8b0000")

def clean_data(input_path: Path, output_path: Path):
    """
    Load, clean, and save the AQI dataset according to defined rules.
    """
    print(f"Loading data from {input_path}...")
    
    # 1. Load with Datetime parsed
    df = pd.read_csv(input_path, parse_dates=['Datetime'])
    
    # 2. Rename Datetime to Date for consistency
    df = df.rename(columns={'Datetime': 'Date'})
    
    # 3. Sort by City then Date
    df = df.sort_values(by=['City', 'Date'])
    
    # 4. Drop duplicate rows for same (City, Date) pairs, keep the first
    df = df.drop_duplicates(subset=['City', 'Date'], keep='first')
    
    # 5. Fix outliers: Cap pollutant values at 99.5th percentile per city
    for pollutant in POLLUTANTS:
        if pollutant in df.columns:
            # Compute 99.5th percentile for each city
            percentiles = df.groupby('City')[pollutant].transform(lambda x: x.quantile(0.995))
            # Cap values exceeding the percentile
            df[pollutant] = np.where((df[pollutant] > percentiles) & pd.notna(df[pollutant]), percentiles, df[pollutant])
            
    # Drop rows where AQI < 0 or AQI > 500
    if 'AQI' in df.columns:
        # Keep rows where AQI is between 0 and 500, or where AQI is NaN (handled later)
        df = df[((df['AQI'] >= 0) & (df['AQI'] <= 500)) | df['AQI'].isna()]
        
    # 7. Handle missing values
    # Set index to Date for time-series interpolation
    df = df.set_index('Date')
    
    # Per-city interpolation (limit=3) then median fill for remaining NaNs
    for pollutant in POLLUTANTS:
        if pollutant in df.columns:
            df[pollutant] = df.groupby('City')[pollutant].transform(
                lambda x: x.interpolate(method='time', limit=3).fillna(x.median())
            )
            
    # Reset index to bring Date back as a column
    df = df.reset_index()
    
    # Drop rows where AQI is still NaN
    if 'AQI' in df.columns:
        df = df.dropna(subset=['AQI'])
        
    # 6. Fix AQI_Bucket: Recalculate based on actual AQI values
    if 'AQI' in df.columns:
        df['AQI_Bucket'] = df['AQI'].apply(lambda x: get_aqi_bucket(x)[0])
        
    # Final sort and index reset
    df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)
    
    # 8. Save to data/cleaned_aqi.csv
    print(f"Saving cleaned data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # 9. Print summary statistics
    print("\n--- Data Cleaning Summary ---")
    print(f"Total Rows: {len(df)}")
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    
    cols_to_describe = [col for col in POLLUTANTS + ['AQI'] if col in df.columns]
    print("\nSummary Statistics for Pollutants and AQI:")
    print(df[cols_to_describe].describe())
    
    if 'AQI_Bucket' in df.columns:
        print("\nAQI Bucket Distribution:")
        print(df['AQI_Bucket'].value_counts(dropna=False))
    print("-----------------------------\n")
    
if __name__ == '__main__':
    clean_data(INPUT_FILE, OUTPUT_FILE)
