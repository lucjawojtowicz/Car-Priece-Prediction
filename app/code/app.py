import dash
from dash import dcc, html, Input, Output, State
import pickle
import pandas as pd
import numpy as np

# ===== wczytaj model, kolumny i wartości domyślne =====
model = pickle.load(open('model/car_price_model.pkl', 'rb'))
model_columns = pickle.load(open('model/model_columns.pkl', 'rb'))
defaults = pickle.load(open('model/defaults.pkl', 'rb'))

# lista marek dostępnych w modelu (wyciągnięta z nazw kolumn brand_XXX)
brand_columns = [c for c in model_columns if c.startswith('brand_')]
brand_options = [{'label': 'Ambassador (default)', 'value': 'Ambassador'}] + \
                [{'label': c.replace('brand_', ''), 'value': c.replace('brand_', '')} for c in brand_columns]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Car Price Predictor"),
    html.P("Enter your car details. If you don't know a value, leave it blank — we'll estimate it for you."),

    html.Label("Year"),
    dcc.Input(id='year', type='number', placeholder=f"e.g. {defaults['year']}"),

    html.Label("Kilometers driven"),
    dcc.Input(id='km_driven', type='number', placeholder=f"e.g. {defaults['km_driven']}"),

    html.Label("Mileage (kmpl)"),
    dcc.Input(id='mileage', type='number', placeholder=f"e.g. {defaults['mileage']}"),

    html.Label("Engine (CC)"),
    dcc.Input(id='engine', type='number', placeholder=f"e.g. {defaults['engine']}"),

    html.Label("Max Power (bhp)"),
    dcc.Input(id='max_power', type='number', placeholder=f"e.g. {defaults['max_power']}"),

    html.Label("Seats"),
    dcc.Input(id='seats', type='number', placeholder=f"e.g. {defaults['seats']}"),

    html.Label("Owner (1=First, 2=Second, 3=Third, 4=Fourth & Above)"),
    dcc.Dropdown(id='owner', options=[
        {'label': 'First Owner', 'value': 1},
        {'label': 'Second Owner', 'value': 2},
        {'label': 'Third Owner', 'value': 3},
        {'label': 'Fourth & Above Owner', 'value': 4},
    ], placeholder="Select owner (optional)"),

    html.Label("Transmission"),
    dcc.Dropdown(id='transmission', options=[
        {'label': 'Manual', 'value': 1},
        {'label': 'Automatic', 'value': 0},
    ], placeholder="Select transmission (optional)"),

    html.Label("Seller Type"),
    dcc.Dropdown(id='seller_type', options=[
        {'label': 'Individual', 'value': 1},
        {'label': 'Dealer', 'value': 0},
        {'label': 'Trustmark Dealer', 'value': 2},
    ], placeholder="Select seller type (optional)"),

    html.Label("Fuel"),
    dcc.Dropdown(id='fuel', options=[
        {'label': 'Diesel', 'value': 'Diesel'},
        {'label': 'Petrol', 'value': 'Petrol'},
    ], placeholder="Select fuel type (optional)"),

    html.Label("Brand"),
    dcc.Dropdown(id='brand', options=brand_options, placeholder="Select brand (optional)"),

    html.Br(),
    html.Button('Predict Price', id='predict-button', n_clicks=0),

    html.Div(id='prediction-output', style={'marginTop': 20, 'fontSize': 24, 'fontWeight': 'bold'})
])


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('year', 'value'),
    State('km_driven', 'value'),
    State('mileage', 'value'),
    State('engine', 'value'),
    State('max_power', 'value'),
    State('seats', 'value'),
    State('owner', 'value'),
    State('transmission', 'value'),
    State('seller_type', 'value'),
    State('fuel', 'value'),
    State('brand', 'value'),
)
def predict_price(n_clicks, year, km_driven, mileage, engine, max_power,
                   seats, owner, transmission, seller_type, fuel, brand):
    if n_clicks == 0:
        return ""

    # ===== stwórz wiersz wejściowy, wypełniony zerami dla wszystkich kolumn modelu =====
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    # ===== wypełnij cechy numeryczne (jeśli puste -> użyj wartości domyślnej/mediany) =====
    input_data['year'] = year if year is not None else defaults['year']
    input_data['km_driven'] = km_driven if km_driven is not None else defaults['km_driven']
    input_data['mileage'] = mileage if mileage is not None else defaults['mileage']
    input_data['engine'] = engine if engine is not None else defaults['engine']
    input_data['max_power'] = max_power if max_power is not None else defaults['max_power']
    input_data['seats'] = seats if seats is not None else defaults['seats']
    input_data['owner'] = owner if owner is not None else defaults['owner']
    input_data['transmission'] = transmission if transmission is not None else defaults['transmission']
    input_data['seller_type'] = seller_type if seller_type is not None else defaults['seller_type']

    # ===== fuel: tylko 'fuel_Petrol' istnieje jako kolumna (Diesel to kategoria referencyjna) =====
    if fuel == 'Petrol':
        input_data['fuel_Petrol'] = 1
    # jeśli fuel == 'Diesel' albo brak wyboru -> zostaje 0 (domyślnie Diesel, najczęstsza kategoria)

    # ===== brand: ustaw odpowiednią kolumnę brand_XXX na 1, jeśli marka istnieje jako kolumna =====
    if brand is not None:
        brand_col = f'brand_{brand}'
        if brand_col in input_data.columns:
            input_data[brand_col] = 1
        # jeśli brand == 'Ambassador' -> zostaje 0 wszędzie (kategoria referencyjna)

    # ===== predykcja =====
    pred_log = model.predict(input_data)
    pred_price = np.exp(pred_log)[0]

    return f"Predicted Selling Price: {pred_price:,.0f}"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)