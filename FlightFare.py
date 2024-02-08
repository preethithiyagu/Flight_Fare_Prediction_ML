import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import re
import datetime

# Load dataset
data = pd.read_csv("a1_FlightFare_Dataset.csv")

def validate_date_format(date_string):
    pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    return bool(pattern.match(date_string))

def set_background():
    st.set_page_config(
        page_title="Flight Fare Prediction",
        page_icon="🛫",
    )

set_background()

st.title("Flight Fare Prediction App")

#User Input
depart_date = st.text_input("Enter departure date (YYYY-MM-DD): ")
if depart_date:  # Check if depart_date is not empty
    if not validate_date_format(depart_date):
        st.write("Invalid date format! Please use the format 'YYYY-MM-DD'.")
depart_place = st.text_input("Enter departure place: ")
arrival_place = st.text_input("Enter arrival place: ")
num_persons = st.number_input("Enter number of persons:", min_value=1, step=1)
    
# Feature selection for demonstration
selected_features = ['Date_of_Journey', 'Source', 'Destination', 'Price']
model_data = data[selected_features].copy()

model_data['Date_of_Journey'] = pd.to_datetime(model_data['Date_of_Journey'], format='%d/%m/%Y')
model_data.loc[:, 'Year'] = model_data['Date_of_Journey'].dt.year
model_data.loc[:, 'Month'] = model_data['Date_of_Journey'].dt.month
model_data.loc[:, 'Day'] = model_data['Date_of_Journey'].dt.day
model_data = model_data.drop(['Date_of_Journey'], axis=1).copy()

# Convert categorical variables into numerical representations
encoder = OneHotEncoder

# Check for missing values in categorical columns
if model_data[['Source', 'Destination']].isnull().any().any():
    st.write("Error: Missing values found in categorical columns. Please handle missing values first.")
else:
    # Ensure columns contain only categorical data
    try:
        encoded_cols = encoder.fit_transform(model_data[['Source', 'Destination']])
        column_names = encoder.get_feature_names_out(['Source', 'Destination'])

        # Check if columns exist in DataFrame
        if not set(['Source', 'Destination']).issubset(model_data.columns):
            st.write("Error: Columns 'Source' and 'Destination' not found in the DataFrame.")
        else:
            model_data_encoded = pd.DataFrame(encoded_cols, columns=column_names)

            # Concatenate encoded columns with original DataFrame
            model_data_encoded = pd.concat([model_data.drop(['Source', 'Destination'], axis=1), model_data_encoded], axis=1)
    except ValueError as e:
        st.write(f"Error: {e}")

# Check available sources and destinations
available_sources = set(data['Source'].unique())
available_destinations = set(data['Destination'].unique())

if depart_place and arrival_place:  # Check if both depart_place and arrival_place are filled
    if (depart_place not in available_sources) or (arrival_place not in available_destinations):
        st.write("No flights available for the provided source and/or destination.")
    else:
        direct_flight_check = len(data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)])
        if direct_flight_check == 0:
            st.write("No direct flights available for the specified route.")
        
    X = model_data.drop('Price', axis=1)
    y = model_data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        st.write(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.write(f"{name} - Test R2 Score: {score}")

    selected_model = models['Random Forest Regressor']

    stops_info = data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)]
    if not stops_info.empty:
        num_stops = stops_info['Total_Stops'].values[0]

        user_input_df = pd.DataFrame({
            'Year': [pd.to_datetime(depart_date).year],
            'Month': [pd.to_datetime(depart_date).month],
            'Day': [pd.to_datetime(depart_date).day],
            'Source': [depart_place],
            'Destination': [arrival_place],
        })

        encoded_user_input = pd.DataFrame(encoder.transform(user_input_df[['Source', 'Destination']]))
        encoded_user_input.columns = encoder.get_feature_names_out(['Source', 'Destination'])  # Updated line
        user_input_df = pd.concat([user_input_df, encoded_user_input], axis=1).drop(['Source', 'Destination'], axis=1)

        base_predicted_fare = selected_model.predict(user_input_df)
        increase_percentage = 0.1
        total_price = base_predicted_fare * num_persons * (1 + increase_percentage)
        Airline = stops_info['Airline'].values[0]  # Assuming 'Airline' is the column containing airline names
        
        if st.button('Reset'):
            depart_date = ''
            depart_place = ''
            arrival_place = ''
            num_persons = 1

    if st.button('Predict'):
        st.write(f"The predicted fare for {num_persons} persons on {Airline} from {depart_place} to {arrival_place} on {depart_date} with {num_stops} stop(s) is: Rs. {total_price}")
