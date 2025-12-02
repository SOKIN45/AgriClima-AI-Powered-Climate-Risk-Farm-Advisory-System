import pandas as pd
import numpy as np
import os


def create_sample_data():
    """
    Create sample climate data if CSV file doesn't exist
    """
    data = [
        {'Month': 'JAN', 'Rainfall_Amount': 97.6, 'Number_of_Rainy_Days': 10, 'Max_Temperature': 29.8,
         'Min_Temperature': 21.7, 'Mean_Temperature': 25.8, 'Dry_Bulb_Temperature': 25.2, 'Wet_Bulb_Temperature': 23.2,
         'Dew_Point': 22.4, 'Vapor_Pressure': 27.2, 'Relative_Humidity': 85, 'Mean_Sea_Level_Pressure': 1010.2,
         'Wind_Direction': 'N', 'Wind_Speed': 2, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 3,
         'Days_with_Lightning': 1},
        {'Month': 'FEB', 'Rainfall_Amount': 86.3, 'Number_of_Rainy_Days': 8, 'Max_Temperature': 30.3,
         'Min_Temperature': 21.6, 'Mean_Temperature': 26.0, 'Dry_Bulb_Temperature': 25.3, 'Wet_Bulb_Temperature': 23.1,
         'Dew_Point': 22.2, 'Vapor_Pressure': 27.0, 'Relative_Humidity': 84, 'Mean_Sea_Level_Pressure': 1010.3,
         'Wind_Direction': 'N', 'Wind_Speed': 2, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 2,
         'Days_with_Lightning': 1},
        {'Month': 'MAR', 'Rainfall_Amount': 57.6, 'Number_of_Rainy_Days': 6, 'Max_Temperature': 31.4,
         'Min_Temperature': 21.9, 'Mean_Temperature': 26.7, 'Dry_Bulb_Temperature': 26.0, 'Wet_Bulb_Temperature': 23.4,
         'Dew_Point': 22.4, 'Vapor_Pressure': 27.2, 'Relative_Humidity': 81, 'Mean_Sea_Level_Pressure': 1010.1,
         'Wind_Direction': 'N', 'Wind_Speed': 2, 'Cloud_Amount': 4, 'Days_with_Thunderstorm': 5,
         'Days_with_Lightning': 2},
        {'Month': 'APR', 'Rainfall_Amount': 62.1, 'Number_of_Rainy_Days': 6, 'Max_Temperature': 32.6,
         'Min_Temperature': 22.7, 'Mean_Temperature': 27.6, 'Dry_Bulb_Temperature': 27.0, 'Wet_Bulb_Temperature': 24.0,
         'Dew_Point': 22.8, 'Vapor_Pressure': 27.9, 'Relative_Humidity': 79, 'Mean_Sea_Level_Pressure': 1009.4,
         'Wind_Direction': 'N', 'Wind_Speed': 2, 'Cloud_Amount': 4, 'Days_with_Thunderstorm': 7,
         'Days_with_Lightning': 4},
        {'Month': 'MAY', 'Rainfall_Amount': 128.9, 'Number_of_Rainy_Days': 11, 'Max_Temperature': 33.0,
         'Min_Temperature': 23.3, 'Mean_Temperature': 28.1, 'Dry_Bulb_Temperature': 27.3, 'Wet_Bulb_Temperature': 24.4,
         'Dew_Point': 23.4, 'Vapor_Pressure': 28.9, 'Relative_Humidity': 80, 'Mean_Sea_Level_Pressure': 1008.9,
         'Wind_Direction': 'N', 'Wind_Speed': 2, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 17,
         'Days_with_Lightning': 10},
        {'Month': 'JUN', 'Rainfall_Amount': 220.1, 'Number_of_Rainy_Days': 16, 'Max_Temperature': 32.1,
         'Min_Temperature': 22.9, 'Mean_Temperature': 27.5, 'Dry_Bulb_Temperature': 26.5, 'Wet_Bulb_Temperature': 24.2,
         'Dew_Point': 23.3, 'Vapor_Pressure': 28.8, 'Relative_Humidity': 83, 'Mean_Sea_Level_Pressure': 1009.0,
         'Wind_Direction': 'S', 'Wind_Speed': 1, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 17,
         'Days_with_Lightning': 9},
        {'Month': 'JUL', 'Rainfall_Amount': 247.3, 'Number_of_Rainy_Days': 17, 'Max_Temperature': 31.7,
         'Min_Temperature': 22.6, 'Mean_Temperature': 27.2, 'Dry_Bulb_Temperature': 26.2, 'Wet_Bulb_Temperature': 24.0,
         'Dew_Point': 23.1, 'Vapor_Pressure': 28.4, 'Relative_Humidity': 84, 'Mean_Sea_Level_Pressure': 1009.0,
         'Wind_Direction': 'S', 'Wind_Speed': 1, 'Cloud_Amount': 6, 'Days_with_Thunderstorm': 16,
         'Days_with_Lightning': 8},
        {'Month': 'AUG', 'Rainfall_Amount': 197.4, 'Number_of_Rainy_Days': 14, 'Max_Temperature': 32.2,
         'Min_Temperature': 22.6, 'Mean_Temperature': 27.4, 'Dry_Bulb_Temperature': 26.4, 'Wet_Bulb_Temperature': 23.9,
         'Dew_Point': 23.0, 'Vapor_Pressure': 28.2, 'Relative_Humidity': 82, 'Mean_Sea_Level_Pressure': 1009.1,
         'Wind_Direction': 'S', 'Wind_Speed': 1, 'Cloud_Amount': 6, 'Days_with_Thunderstorm': 13,
         'Days_with_Lightning': 8},
        {'Month': 'SEP', 'Rainfall_Amount': 220.8, 'Number_of_Rainy_Days': 15, 'Max_Temperature': 32.1,
         'Min_Temperature': 22.5, 'Mean_Temperature': 27.3, 'Dry_Bulb_Temperature': 26.3, 'Wet_Bulb_Temperature': 23.9,
         'Dew_Point': 23.0, 'Vapor_Pressure': 28.2, 'Relative_Humidity': 83, 'Mean_Sea_Level_Pressure': 1009.4,
         'Wind_Direction': 'S', 'Wind_Speed': 2, 'Cloud_Amount': 6, 'Days_with_Thunderstorm': 15,
         'Days_with_Lightning': 8},
        {'Month': 'OCT', 'Rainfall_Amount': 191.6, 'Number_of_Rainy_Days': 14, 'Max_Temperature': 31.5,
         'Min_Temperature': 22.4, 'Mean_Temperature': 27.0, 'Dry_Bulb_Temperature': 26.1, 'Wet_Bulb_Temperature': 23.9,
         'Dew_Point': 23.1, 'Vapor_Pressure': 28.4, 'Relative_Humidity': 84, 'Mean_Sea_Level_Pressure': 1009.1,
         'Wind_Direction': 'S', 'Wind_Speed': 2, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 16,
         'Days_with_Lightning': 11},
        {'Month': 'NOV', 'Rainfall_Amount': 127.1, 'Number_of_Rainy_Days': 10, 'Max_Temperature': 31.1,
         'Min_Temperature': 22.2, 'Mean_Temperature': 26.7, 'Dry_Bulb_Temperature': 26.0, 'Wet_Bulb_Temperature': 23.9,
         'Dew_Point': 23.0, 'Vapor_Pressure': 28.3, 'Relative_Humidity': 84, 'Mean_Sea_Level_Pressure': 1008.8,
         'Wind_Direction': 'S', 'Wind_Speed': 2, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 10,
         'Days_with_Lightning': 7},
        {'Month': 'DEC', 'Rainfall_Amount': 137.5, 'Number_of_Rainy_Days': 9, 'Max_Temperature': 30.4,
         'Min_Temperature': 22.1, 'Mean_Temperature': 26.3, 'Dry_Bulb_Temperature': 25.7, 'Wet_Bulb_Temperature': 23.6,
         'Dew_Point': 22.9, 'Vapor_Pressure': 28.0, 'Relative_Humidity': 85, 'Mean_Sea_Level_Pressure': 1009.2,
         'Wind_Direction': 'S', 'Wind_Speed': 2, 'Cloud_Amount': 5, 'Days_with_Thunderstorm': 5,
         'Days_with_Lightning': 4}
    ]

    return pd.DataFrame(data)


def load_climate_data(csv_path):
    """
    Load climate data from CSV file, create sample data if file doesn't exist
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded data from: {csv_path}")
    except FileNotFoundError:
        print(f"⚠️ File {csv_path} not found. Creating sample data...")
        df = create_sample_data()

        # Create directory and save the sample data
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ Sample data saved to: {csv_path}")

    return df


def clean_and_prepare_data(df):
    """
    Clean and prepare the climate data for modeling
    """
    print(f"📊 Cleaning data... Original shape: {df.shape}")

    # Convert wind direction to numerical values (angles)
    wind_direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    df['Wind_Direction_Num'] = df['Wind_Direction'].map(wind_direction_map)

    # Ensure all numeric columns are properly typed
    numeric_columns = [
        'Rainfall_Amount', 'Number_of_Rainy_Days', 'Max_Temperature',
        'Min_Temperature', 'Mean_Temperature', 'Dry_Bulb_Temperature',
        'Wet_Bulb_Temperature', 'Dew_Point', 'Vapor_Pressure',
        'Relative_Humidity', 'Mean_Sea_Level_Pressure', 'Wind_Speed',
        'Cloud_Amount', 'Days_with_Thunderstorm', 'Days_with_Lightning'
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"✅ Data cleaned. New shape: {df.shape}")
    return df


def get_data_summary(df):
    """
    Get summary statistics of the dataset
    """
    summary = {
        'total_months': len(df),
        'annual_rainfall': df['Rainfall_Amount'].sum(),
        'avg_temperature': df['Mean_Temperature'].mean(),
        'avg_rainfall': df['Rainfall_Amount'].mean(),
        'avg_humidity': df['Relative_Humidity'].mean(),
        'max_rainfall_month': df.loc[df['Rainfall_Amount'].idxmax(), 'Month'],
        'min_rainfall_month': df.loc[df['Rainfall_Amount'].idxmin(), 'Month']
    }
    return summary


# Main execution for testing
if __name__ == "__main__":
    print("=" * 50)
    print("🌦️ CLIMATE DATA PROCESSING")
    print("=" * 50)

    # Get current working directory
    current_dir = os.getcwd()
    print(f"📁 Current working directory: {current_dir}")

    # Define paths
    raw_data_path = "data/raw/climate_data.csv"
    processed_data_path = "data/processed/climate_data_clean.csv"

    # Convert to absolute paths for clarity
    raw_data_abs = os.path.join(current_dir, raw_data_path)
    processed_data_abs = os.path.join(current_dir, processed_data_path)

    print(f"📂 Raw data path: {raw_data_abs}")
    print(f"📂 Processed data path: {processed_data_abs}")

    # Step 1: Load data
    print("\n" + "=" * 50)
    print("1️⃣ LOADING DATA")
    print("=" * 50)
    df = load_climate_data(raw_data_path)

    # Step 2: Clean data
    print("\n" + "=" * 50)
    print("2️⃣ CLEANING DATA")
    print("=" * 50)
    df_clean = clean_and_prepare_data(df)

    # Step 3: Get summary
    print("\n" + "=" * 50)
    print("3️⃣ DATA SUMMARY")
    print("=" * 50)
    summary = get_data_summary(df_clean)

    print(f"📊 Dataset shape: {df_clean.shape}")
    print(f"🌧️ Annual Rainfall: {summary['annual_rainfall']:.1f} mm")
    print(f"🌡️ Average Temperature: {summary['avg_temperature']:.1f}°C")
    print(f"💧 Average Humidity: {summary['avg_humidity']:.1f}%")
    print(f"☔ Wettest Month: {summary['max_rainfall_month']}")
    print(f"☀️ Driest Month: {summary['min_rainfall_month']}")

    # Step 4: Save cleaned data
    print("\n" + "=" * 50)
    print("4️⃣ SAVING PROCESSED DATA")
    print("=" * 50)

    # Create directory if it doesn't exist
    processed_dir = os.path.dirname(processed_data_abs)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"✅ Created directory: {processed_dir}")

    # Save the file
    df_clean.to_csv(processed_data_abs, index=False)

    # Verify the file was saved
    if os.path.exists(processed_data_abs):
        file_size = os.path.getsize(processed_data_abs)
        print(f"✅ Cleaned data saved successfully!")
        print(f"📁 File location: {processed_data_abs}")
        print(f"📏 File size: {file_size} bytes")
        print(f"📄 File preview:")
        print(df_clean.head())
    else:
        print(f"❌ ERROR: File was not saved!")
        print(f"Check write permissions in: {processed_dir}")

    print("\n" + "=" * 50)
    print("✅ PROCESSING COMPLETE")
    print("=" * 50)