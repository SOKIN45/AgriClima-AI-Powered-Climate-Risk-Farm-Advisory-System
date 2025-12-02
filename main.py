import os
import sys
from src.data_processing import load_climate_data, clean_and_prepare_data, get_data_summary
from src.feature_engineering import create_features, prepare_model_data
from src.model_training import ClimateModel
from src.visualization import ClimateVisualizer
import matplotlib.pyplot as plt


def main():
    print("Starting Climate Prediction Project...")

    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Data Loading
    print("1. Loading data from CSV...")
    df = load_climate_data("data/raw/climate_data.csv")

    # Step 2: Data Cleaning
    print("2. Cleaning and preparing data...")
    df_clean = clean_and_prepare_data(df)

    # Step 3: Feature Engineering
    print("3. Creating features...")
    df_features = create_features(df_clean)

    # Display summary
    summary = get_data_summary(df_features)
    print(f"\nDataset Summary:")
    print(f"  Total months: {summary['total_months']}")
    print(f"  Annual Rainfall: {summary['annual_rainfall']:.1f} mm")
    print(f"  Average Temperature: {summary['avg_temperature']:.1f}°C")
    print(f"  Wettest Month: {summary['max_rainfall_month']}")
    print(f"  Driest Month: {summary['min_rainfall_month']}")

    # Step 4: Model Training
    print("\n4. Training machine learning models...")
    X, y_rainfall, y_temperature, feature_names = prepare_model_data(df_features)

    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")

    model = ClimateModel()
    rainfall_model, temperature_model = model.train_models(X, y_rainfall, y_temperature)

    # Step 5: Save Models
    print("5. Saving trained models...")
    model.save_models("models/climate_models.joblib")

    # Step 6: Create Visualizations
    print("6. Generating visualizations...")
    viz = ClimateVisualizer(df_features)

    # Monthly trends
    fig_trends = viz.plot_monthly_trends()
    fig_trends.savefig("outputs/monthly_trends.png", dpi=300, bbox_inches='tight')

    # Interactive plots
    fig1, fig2 = viz.create_interactive_plots()
    fig1.write_html("outputs/temp_rain_correlation.html")
    fig2.write_html("outputs/monthly_patterns.html")

    # Feature importance
    if hasattr(rainfall_model, 'feature_importances_'):
        fig_importance = viz.plot_feature_importance(rainfall_model, feature_names)
        fig_importance.savefig("outputs/feature_importance.png", dpi=300, bbox_inches='tight')

    print("\nProject execution completed successfully!")
    print("\nGenerated Files:")
    print("  - data/processed/climate_data_clean.csv")
    print("  - models/climate_models.joblib")
    print("  - outputs/monthly_trends.png")
    print("  - outputs/feature_importance.png")
    print("  - outputs/temp_rain_correlation.html")
    print("  - outputs/monthly_patterns.html")

    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app/streamlit_app.py")
    print("2. Check the outputs/ folder for generated visualizations")


if __name__ == "__main__":
    main()