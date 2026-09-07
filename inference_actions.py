def generate_hvac_action(predicted_heating, predicted_cooling, current_co2, co2_max_acceptable=1000):
    """
    Translates raw numerical load predictions into actionable HVAC optimizations.
    """
    actions = []

    # 1. Air Quality
    if current_co2 >= co2_max_acceptable:
        actions.append(
            f"CRITICAL IAQ: CO2 levels at {current_co2} ppm. Increase outdoor air damper by 15% to flush stale air.")

    # 2. Load Efficiency
    # Using 23.66 and 18.95 median thresholds
    if predicted_cooling > predicted_heating and predicted_cooling >= 23.66:
        actions.append(
            f"OPTIMIZATION: High cooling load predicted ({predicted_cooling:.2f}). Pre-cool the building during off-peak hours and engage economizer.")

    elif predicted_heating > predicted_cooling and predicted_heating >= 18.95:
        actions.append(
            f"OPTIMIZATION: High heating load predicted ({predicted_heating:.2f}). Optimize boiler staging and verify thermal envelope seals.")

    else:
        actions.append("STATUS: Optimal thermodynamic balance predicted. Maintain current scheduled setpoints.")

    return actions

# --- Example Usage ---
# output = generate_hvac_action(predicted_heating=12.5, predicted_cooling=26.8, current_co2=1150)
# for action in output:
#     print(action)