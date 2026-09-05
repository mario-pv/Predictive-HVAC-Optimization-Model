import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split


def train_and_evaluate():
    # Load preprocessed dataset
    df = pd.read_csv("preprocessed_energy_efficiency.csv")

    X = df.drop(columns=["Heating_Load", "Cooling_Load"])
    y = df[["Heating_Load", "Cooling_Load"]]

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # B2: Build the Random Forest algorithm
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)