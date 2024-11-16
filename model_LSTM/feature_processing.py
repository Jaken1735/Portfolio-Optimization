"""
Calculate the relevant features for each stock in the S&P 500
"""

import pandas as pd

def calculate_features(data_dict):
    """
    Calculate technical indicators for each asset.

    Parameters:
        data_dict (dict): Dictionary with stock symbols as keys and DataFrames as values.

    Returns:
        dict: Dictionary with updated DataFrames containing technical indicators.
    """
    feature_data_dict = {}
    for symbol, data in data_dict.items():
        df = data.copy()
        if 'Close' in df.columns:
            # Moving Averages
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()

            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

            # Momentum
            df['Momentum'] = df['Close'].diff(10)

            feature_data_dict[symbol] = df
        else:
            print(f"Skipping {symbol}: 'Close' column not found.")
    return feature_data_dict
