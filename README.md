# Concrete Strength ML

This project builds and evaluates a machine learning model to predict the compressive strength of concrete from its mix proportions and curing age. It uses a Random Forest regressor trained on the classic concrete compressive strength dataset (cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age). 

## Flask web app
This project includes a small Flask web application that wraps the trained Random Forest model and provides a clean UI for entering concrete mix proportions and viewing the predicted compressive strength along with suggested civil engineering applications for that strength level. 

### Run locally
Make sure Python and the required libraries (Flask, scikit-learn, joblib, etc.) are installed, then from the project root run:
From the project root:
Since Anaconda prompt and built using Jupyter Notebook is used the following are the steps to proceed:

1: Open terminal Anaconda and paste the below commands
```bash
 cd C:\Users\USERNAME\Downloads\concrete_strength 
```
Note: ("Downloads" because the file is saved in it)

2:
```bash
python app.py
```
3: Once the server is running it will show the URL directing to the page it shows 
Running on  
```bash
http://127.0.0.1:5000
```
4: Click the URL and the "Concrete Strength Predictor" page will be opened in the browser
