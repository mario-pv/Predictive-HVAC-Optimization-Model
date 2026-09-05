import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


def optimize_model():
    df = pd.read_csv("preprocessed_energy_efficiency.csv")
    X = df.drop(columns=["Heating_Load", "Cooling_Load"])
    y = df[["Heating_Load", "Cooling_Load"]]

    # B5: Apply Cross-Validation
    print("Applying 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_base = RandomForestRegressor(random_state=42)

    cv_scores = cross_val_score(
        rf_base, X, y, cv=kf, scoring="r2", n_jobs=-1
    )
    print(f"5-Fold CV R^2 Scores: {cv_scores}")
    print(f"Mean CV R^2 Score:    {cv_scores.mean():.4f}")