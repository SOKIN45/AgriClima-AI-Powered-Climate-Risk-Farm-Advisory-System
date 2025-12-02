import pandas as pd
import numpy as np


def create_features(df):
    """
    Create additional features for the model
    """
    # Create Month_Num if it doesn't exist
    if 'Month_Num' not in df.columns and 'Month' in df.columns:
        # Convert month names to numerical values
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        df['Month_Num'] = df['Month'].map(month_map)

    # Seasonal features
    df['Season'] = df['Month_Num'].apply(lambda x:
                                         'Dry' if x in [1, 2, 3, 4] else 'Wet')

    # Create seasonal indicators
    seasons = {
        1: 'Dry', 2: 'Dry', 3: 'Dry', 4: 'Dry',
        5: 'Wet', 6: 'Wet', 7: 'Wet', 8: 'Wet',
        9: 'Wet', 10: 'Wet', 11: 'Wet', 12: 'Dry'
    }
    df['Season_Label'] = df['Month_Num'].map(seasons)

    # Create cyclical features for months
    df['Month_sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)

    # Temperature range
    df['Temperature_Range'] = df['Max_Temperature'] - df['Min_Temperature']

    # Create lag features (for time series analysis)
    for lag in [1, 2, 3]:
        df[f'Rainfall_Lag_{lag}'] = df['Rainfall_Amount'].shift(lag)
        df[f'Temp_Lag_{lag}'] = df['Mean_Temperature'].shift(lag)

    # Fill NaN values from lag features
    df = df.fillna(method='bfill')

    return df


def prepare_model_data(df):
    """
    Prepare features and targets for machine learning
    """
    # Ensure Month_Num exists
    if 'Month_Num' not in df.columns and 'Month' in df.columns:
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        df['Month_Num'] = df['Month'].map(month_map)

    # Feature columns
    feature_columns = [
        'Month_Num', 'Month_sin', 'Month_cos',
        'Number_of_Rainy_Days', 'Max_Temperature', 'Min_Temperature',
        'Dry_Bulb_Temperature', 'Wet_Bulb_Temperature', 'Dew_Point',
        'Vapor_Pressure', 'Relative_Humidity', 'Mean_Sea_Level_Pressure',
        'Wind_Direction_Num', 'Wind_Speed', 'Cloud_Amount',
        'Days_with_Thunderstorm', 'Days_with_Lightning',
        'Temperature_Range', 'Rainfall_Lag_1', 'Rainfall_Lag_2',
        'Rainfall_Lag_3', 'Temp_Lag_1', 'Temp_Lag_2', 'Temp_Lag_3'
    ]

    # Target variables
    rainfall_target = 'Rainfall_Amount'
    temperature_target = 'Mean_Temperature'

    X = df[feature_columns]
    y_rainfall = df[rainfall_target]
    y_temperature = df[temperature_target]

    return X, y_rainfall, y_temperature, feature_columns