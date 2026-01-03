from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained RandomForest model (13 input features)
rf = joblib.load("concrete_rf_model.joblib")


def build_mix(cement, slag, ash, water, superplastic, coarseagg, fineagg, age):
    """Create a one-row DataFrame with raw + engineered features."""
    water_cement_ratio = water / cement
    total_binder = cement + slag + ash
    aggregate_to_cement = (coarseagg + fineagg) / cement
    cement_water_interaction = cement * water
    age_strength_proxy = (age ** 0.5) * cement

    return pd.DataFrame([{
        "cement": cement,
        "slag": slag,
        "ash": ash,
        "water": water,
        "superplastic": superplastic,
        "coarseagg": coarseagg,
        "fineagg": fineagg,
        "age": age,
        "water_cement_ratio": water_cement_ratio,
        "total_binder": total_binder,
        "aggregate_to_cement": aggregate_to_cement,
        "cement_water_interaction": cement_water_interaction,
        "age_strength_proxy": age_strength_proxy,
    }])


def describe_applications(strength_mpa):
    """Map predicted strength to label + list of civil applications."""
    s = strength_mpa
    if s < 20:
        label = "Lean / non-structural concrete"
        uses = [
            "Blinding layers and levelling concrete",
            "Simple pathways and non-traffic slabs",
            "Non-structural infill or mass concrete"
        ]
    elif s < 30:
        label = "Normal structural concrete (approx. M20–M25)"
        uses = [
            "Residential slabs, beams and columns",
            "Light commercial buildings",
            "Low-rise frames and shallow footings"
        ]
    elif s < 50:
        label = "Medium–higher strength concrete (approx. M30–M40)"
        uses = [
            "Heavier residential/commercial structures",
            "Parking decks and industrial floors",
            "Retaining walls and water-retaining structures"
        ]
    elif s < 70:
        label = "High-strength concrete (approx. M50–M60)"
        uses = [
            "High-rise building columns and core walls",
            "Long-span beams and bridge elements",
            "Structures needing smaller sections with high loads"
        ]
    else:
        label = "Very high / ultra‑high strength concrete"
        uses = [
            "Special bridges and high-rise mega columns",
            "Precast/prestressed elements",
            "Infrastructure requiring very high durability and strength"
        ]
    return label, uses


# ---------- Routes ----------

@app.route("/", methods=["GET"])
def home():
    # Just render the empty form
    return render_template("index.html")


@app.route("/predict_form", methods=["POST"])
def predict_form():
    # Read form values
    form = request.form
    cement = float(form["cement"])
    slag = float(form["slag"])
    ash = float(form["ash"])
    water = float(form["water"])
    superplastic = float(form["superplastic"])
    coarseagg = float(form["coarseagg"])
    fineagg = float(form["fineagg"])
    age = float(form["age"])

    # Build feature row and predict
    new_mix = build_mix(cement, slag, ash, water,
                        superplastic, coarseagg, fineagg, age)
    pred = float(rf.predict(new_mix)[0])

    # Map strength to applications
    strength_label, strength_uses = describe_applications(pred)

    # Re-render page with results
    return render_template(
        "index.html",
        prediction=round(pred, 2),
        strength_label=strength_label,
        strength_uses=strength_uses
    )


@app.route("/predict", methods=["POST"])
def predict_json():
    # JSON API endpoint (kept for programmatic access)
    data = request.get_json()
    cement = float(data["cement"])
    slag = float(data["slag"])
    ash = float(data["ash"])
    water = float(data["water"])
    superplastic = float(data["superplastic"])
    coarseagg = float(data["coarseagg"])
    fineagg = float(data["fineagg"])
    age = float(data["age"])

    new_mix = build_mix(cement, slag, ash, water,
                        superplastic, coarseagg, fineagg, age)
    pred = float(rf.predict(new_mix)[0])

    return jsonify({"predicted_strength_MPa": pred})


if __name__ == "__main__":
    app.run(debug=True)
