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

    # B6: Hyperparameter Tuning via GridSearchCV
    print("\nExecuting GridSearchCV Hyperparameter Tuning...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X, y)

    print("\nOptimal Hyperparameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nOptimized Model R^2 Score: {grid_search.best_score_:.4f}")


if __name__ == "__main__":
    optimize_model()