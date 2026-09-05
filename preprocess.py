import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def run_preprocessing():
    print("Fetching UCI Energy Efficiency Dataset (ID: 242)...")
    # Fetch dataset using ucimlrepo library or fallback direct load
    try:
        energy_efficiency = fetch_ucirepo(id=242)
        X = energy_efficiency.data.features
        y = energy_efficiency.data.targets
        df = pd.concat([X, y], axis=1)
    except Exception as e:
        print(
            f"ucimlrepo fetch failed: {e}. Attempting direct CSV load from web..."
        )
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.csv"
        df = pd.read_csv(url)

    # Clean empty/unnamed columns if present
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Standardize column names
    feature_names = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
    ]
    target_names = ["Heating_Load", "Cooling_Load"]

    df.columns = feature_names + target_names

    # Check missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    df = df.dropna()

    # Scale features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_names])
    df_scaled = pd.DataFrame(
        scaled_features, columns=[f"{col}_Scaled" for col in feature_names]
    )

    # Combine scaled features with original target variables
    preprocessed_df = pd.concat([df_scaled, df[target_names]], axis=1)

    # Save output to CSV
    output_filename = "preprocessed_energy_efficiency.csv"
    preprocessed_df.to_csv(output_filename, index=False)
    print(
        f"\nPreprocessing complete! Saved preprocessed dataset to '{output_filename}' ({preprocessed_df.shape[0]} rows, {preprocessed_df.shape[1]} columns)."
    )


if __name__ == "__main__":
    run_preprocessing()