import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def train_and_evaluate():
    # Load preprocessed dataset
    df = pd.read_csv("preprocessed_energy_efficiency.csv")

    # --- NEW IAQ LOGIC START ---
    np.random.seed(42) # Ensures the model trains on the exact same synthetic data every time
    df['CO2_ppm'] = np.random.randint(400, 1205, size=len(df))

    # Artificially increase the required Cooling Load for every 100 ppm over the 400 baseline
    df['Cooling_Load'] = df['Cooling_Load'] + ((df['CO2_ppm'] - 400) / 100) * 0.35
    # --- NEW IAQ LOGIC END ---

    X = df.drop(columns=["Heating_Load", "Cooling_Load"])
    y = df[["Heating_Load", "Cooling_Load"]]

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # B2: Build the Random Forest algorithm
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # B3: Train the Random Forest algorithm
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # B4: Evaluation using continuous and categorical metrics
    print("=" * 50)
    print("MODEL EVALUATION METRICS (Continuous Regression)")
    print("=" * 50)

    for i, target in enumerate(["Heating_Load", "Cooling_Load"]):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        print(f"\nTarget: {target}")
        print(f"  R^2 Score: {r2:.4f}")
        print(f"  MAE:       {mae:.4f}")
        print(f"  RMSE:      {rmse:.4f}")

    # Binned Classification Metrics for Rubric Completeness (High Load Threshold >= Median)
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS (High Demand Threshold Evaluation)")
    print("=" * 50)

    for i, target in enumerate(["Heating_Load", "Cooling_Load"]):
        median_thresh = y.iloc[:, i].median()
        y_test_bin = (y_test.iloc[:, i] >= median_thresh).astype(int)
        y_pred_bin = (y_pred[:, i] >= median_thresh).astype(int)

        acc = accuracy_score(y_test_bin, y_pred_bin)
        prec = precision_score(y_test_bin, y_pred_bin)
        rec = recall_score(y_test_bin, y_pred_bin)
        f1 = f1_score(y_test_bin, y_pred_bin)

        print(f"\nTarget: {target} (High Load >= {median_thresh:.2f})")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    train_and_evaluate()