# Concrete Strength ML

This project builds and evaluates a machine learning model to predict the compressive strength of concrete from its mix proportions and curing age. It uses a Random Forest regressor trained on the classic concrete compressive strength dataset (cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age). 

## Flask web app
This project includes a small Flask web application that wraps the trained Random Forest model and provides a clean UI for entering concrete mix proportions and viewing the predicted compressive strength along with suggested civil engineering applications for that strength level. 

### Run locally
Make sure Python and the required libraries (Flask, scikit-learn, joblib, etc.) are installed, then from the project root run:
From the project root:
```bash
python app.py
```
Once the server is running, open this URL in your browser to use the web UI:
Concrete Strength Predictor - http://127.0.0.1:5000
