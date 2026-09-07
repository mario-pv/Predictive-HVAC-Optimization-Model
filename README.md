# Predictive HVAC Optimization Model

An advanced machine learning decision support system designed to optimize commercial building climate control. This project utilizes a Random Forest regression algorithm to analyze structural building thermodynamics, dynamic external weather data, and indoor air quality metrics, predicting precise heating and cooling loads to minimize energy waste.

## Tech Stack
* **Language:** Python 3.10+
* **Machine Learning:** Scikit-learn (RandomForestRegressor)
* **Data Processing:** Pandas, NumPy
* **Data Ingestion:** Requests, JSON (REST API integration)

## System Architecture & Data Pipeline
This model moves beyond static HVAC schedules by creating a dynamic, data-driven pipeline:
1. **Data Aggregation:** Ingests 768 structural building configurations from the UCI Machine Learning Repository (features include Relative Compactness, Surface Area, Glazing Distribution).
2. **Dynamic API Integration:** Executes HTTP GET requests to the OpenWeatherMap REST API to fetch live external temperature and humidity, merging it with the structural data.
3. **Feature Engineering:** Synthesizes an Indoor Air Quality (IAQ) metric by generating a CO2 (ppm) feature, calculating the thermodynamic energy penalty required for mechanical ventilation.
4. **Preprocessing:** Cleans missing values and applies scikit-learn's `StandardScaler` for uniform feature scaling prior to model inference.

## Model Performance
The ensemble model was rigorously tested using cross-validation and hyperparameter tuning to ensure stable predictions, significantly exceeding the baseline business goal of an 0.85 R-squared score.

| **Target Variable** | **R² Score** | **RMSE** | **MAE** |
| :--- | :--- | :--- | :--- |
| Heating Load | 0.9969 | 0.5656 | 0.4126 |
| Cooling Load | 0.9528 | 2.1104 | 1.3505 |

## Actionable Optimization Directives
To transition the model from purely predictive to highly prescriptive, a post-processing algorithm evaluates the numerical output against established median demand thresholds. The system outputs concrete, actionable directives for facility managers or programmable logic controllers (PLCs).

```text
==================================================
ACTIONABLE HVAC OPTIMIZATIONS (Live Run)
==================================================
Predicted Heating Load: 16.57
Predicted Cooling Load: 17.42
Current CO2 Level: 408.0 ppm

-> STATUS: Optimal thermodynamic balance predicted. Maintain current scheduled setpoints.