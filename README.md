# A1: Predicting Car Price

**Student:** [TLucja Wojtowicz st127262]
**Course:** AT82.03: Machine Learning

Machine Learning assignment (AT82.03) - a web-based car price prediction system 

## Project Structure
Assignment1/
├── Predicting_car_price.ipynb 
├── Cars.csv 
├── README.md
└── app/
    ├── Dockerfile
    ├── docker-compose.yaml
    └── code/
        ├── app.py  
        ├── requirements.txt
        └── model/
            ├── car_price_model.pkl
            ├── model_columns.pkl
            └── defaults.pkl


## Task 1 & 2: Notebook

The notebook (`Predicting_car_price.ipynb`) covers:
- Data loading and cleaning (handling missing values, duplicates, unit conversion)
- Exploratory Data Analysis (distributions, correlations, feature relationships)
- Feature engineering (owner mapping, fuel filtering, brand extraction, etc.)
- Model comparison (Linear Regression, Random Forest, SVR) via cross-validation
- Hyperparameter tuning with GridSearchCV
- Final evaluation on a held-out test set
- Feature importance analysis
- A written summary and analysis of results

## Task 3: Web Application

A Dash-based web app that predicts a car's selling price based on user input.
Users can leave fields blank — missing values are automatically filled in
using median/mode values from the training data.

### Running locally (without Docker)

```bash
cd app/code
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8050` in your browser.

### Running with Docker

```bash
cd app
docker-compose up --build
```

Then open `http://localhost:8050` in your browser.

## Model

The final model is a Random Forest Regressor, tuned via GridSearchCV, achieving
an R² of ~0.94 on the held-out test set. See the notebook for full details on
feature importance and model comparison.

