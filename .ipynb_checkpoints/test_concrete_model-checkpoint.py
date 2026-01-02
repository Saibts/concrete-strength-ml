import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import joblib
import pandas as pd
import os

# 1. Load trained model
rf = joblib.load("concrete_rf_model.joblib")

print("Enter concrete mix details:")

cement = float(input("Cement (kg/m^3): "))
slag = float(input("Slag (kg/m^3): "))
ash = float(input("Fly ash (kg/m^3): "))
water = float(input("Water (kg/m^3): "))
superplastic = float(input("Superplasticizer (kg/m^3): "))
coarseagg = float(input("Coarse aggregate (kg/m^3): "))
fineagg = float(input("Fine aggregate (kg/m^3): "))
age = float(input("Age (days): "))

# 2. Compute engineered features (same as notebook)
water_cement_ratio = water / cement
total_binder = cement + slag + ash
aggregate_to_cement = (coarseagg + fineagg) / cement
cement_water_interaction = cement * water
age_strength_proxy = (age ** 0.5) * cement

# 3. Build DataFrame for this one input
new_mix = pd.DataFrame([{
    'cement': cement,
    'slag': slag,
    'ash': ash,
    'water': water,
    'superplastic': superplastic,
    'coarseagg': coarseagg,
    'fineagg': fineagg,
    'age': age,
    'water_cement_ratio': water_cement_ratio,
    'total_binder': total_binder,
    'aggregate_to_cement': aggregate_to_cement,
    'cement_water_interaction': cement_water_interaction,
    'age_strength_proxy': age_strength_proxy
}])

# 4. Predict
pred_strength = rf.predict(new_mix)[0]
print("\nPredicted compressive strength (MPa):", pred_strength)

# 5. Append input + prediction to a CSV log
log_row = new_mix.copy()
log_row["predicted_strength_MPa"] = pred_strength

log_file = "concrete_predictions_log.csv"

if os.path.exists(log_file):
    # Append without header
    log_row.to_csv(log_file, mode="a", index=False, header=False)
else:
    # Create new file with header
    log_row.to_csv(log_file, mode="w", index=False, header=True)

print(f"\nSaved this test case to {log_file}")
