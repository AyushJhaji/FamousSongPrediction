import pandas as pd
import numpy as np
import re

def clean_spotify_data():
    """Clean the corrupted Spotify dataset"""
    
    # Load data
    df = pd.read_csv('spotify-2023.csv', encoding='latin1')
    print(f"Original dataset: {len(df)} rows")
    
    # Function to clean streams column
    def clean_streams(value):
        if pd.isna(value):
            return np.nan
        
        str_val = str(value).strip()
        
        # If it's already a clean number
        if str_val.replace('.', '').isdigit() and len(str_val) <= 12:
            try:
                return int(float(str_val))
            except:
                pass
        
        # Handle concatenated numbers
        if len(str_val) > 15 and str_val.isdigit():
            # Extract reasonable stream counts (8-10 digits typically)
            for length in [9, 10, 8, 11]:
                if len(str_val) >= length:
                    candidate = str_val[:length]
                    num = int(candidate)
                    if 1000 <= num <= 5000000000:  # Reasonable range
                        return num
        
        # Extract first reasonable number
        numbers = re.findall(r'\d+', str_val)
        for num_str in numbers:
            if 4 <= len(num_str) <= 12:
                try:
                    num = int(num_str)
                    if 1000 <= num <= 5000000000:
                        return num
                except:
                    continue
        
        return np.nan
    
    # Clean streams column
    print("Cleaning streams data...")
    df['streams_clean'] = df['streams'].apply(clean_streams)
    
    # Remove rows with invalid streams
    df_clean = df.dropna(subset=['streams_clean']).copy()
    df_clean['streams'] = df_clean['streams_clean'].astype(int)
    df_clean = df_clean.drop('streams_clean', axis=1)
    
    print(f"Cleaned dataset: {len(df_clean)} rows ({len(df) - len(df_clean)} rows removed)")
    
    # Save cleaned data
    df_clean.to_csv('spotify-2023-cleaned.csv', index=False)
    print("Cleaned data saved as 'spotify-2023-cleaned.csv'")
    
    return df_clean

if __name__ == "__main__":
    clean_spotify_data()
