import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import re

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

# --- Initialize session state variables ---
if 'depart_date' not in st.session_state:
    st.session_state.depart_date = ''

if 'depart_place' not in st.session_state:
    st.session_state.depart_place = ''

if 'arrival_place' not in st.session_state:
    st.session_state.arrival_place = ''

if 'num_persons' not in st.session_state:
    st.session_state.num_persons = 1

# --- User Inputs with session state ---
depart_date = st.text_input("Enter departure date (YYYY-MM-DD): ", value=st.session_state.depart_date)
depart_place = st.text_input("Enter departure place: ", value=st.session_state.depart_place)
arrival_place = st.text_input("Enter arrival place: ", value=st.session_state.arrival_place)
num_persons = st.number_input("Enter number of persons:", min_value=1, step=1, value=st.session_state.num_persons)

# Update session state after inputs (keeps values persistent)
st.session_state.depart_date = depart_date
st.session_state.depart_place = depart_place
st.session_state.arrival_place = arrival_place
st.session_state.num_persons = num_persons

# Validate date format
if depart_date and not validate_date_format(depart_date):
    st.error("Invalid date format! Please use the format 'YYYY-MM-DD'.")

# Prepare data for model
selected_features = ['Date_of_Journey', 'Source', 'Destination', 'Price']
model_data = data[selected_features].copy()

model_data['Date_of_Journey'] = pd.to_datetime(model_data['Date_of_Journey'], format='%d/%m/%Y')
model_data['Year'] = model_data['Date_of_Journey'].dt.year
model_data['Month'] = model_data['Date_of_Journey'].dt.month
model_data['Day'] = model_data['Date_of_Journey'].dt.day
model_data = model_data.drop(['Date_of_Journey'], axis=1)

encoder = OneHotEncoder()

if model_data[['Source', 'Destination']].isnull().any().any():
    st.error("Missing values found in categorical columns. Please handle missing values first.")
else:
    try:
        encoded_cols = encoder.fit_transform(model_data[['Source', 'Destination']])
        column_names = encoder.get_feature_names_out(['Source', 'Destination'])
        model_data_encoded = pd.DataFrame(encoded_cols.toarray(), columns=column_names)
        model_data_encoded = pd.concat([model_data.drop(['Source', 'Destination'], axis=1), model_data_encoded], axis=1)
    except ValueError as e:
        st.error(f"Error encoding categorical features: {e}")

# Available sources and destinations
available_sources = set(data['Source'].unique())
available_destinations = set(data['Destination'].unique())

# Prediction logic
if depart_date and depart_place and arrival_place and num_persons >= 1:

    if depart_place not in available_sources or arrival_place not in available_destinations:
        st.warning("No flights available for the provided source and/or destination.")
    else:
        direct_flight_check = len(data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)])
        if direct_flight_check == 0:
            st.warning("No direct flights available for the specified route.")

        X = model_data_encoded.drop('Price', axis=1)
        y = model_data['Price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(random_state=42),
            'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

        selected_model = models['Random Forest Regressor']

        stops_info = data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)]
        if not stops_info.empty:
            num_stops = stops_info['Total_Stops'].values[0]
            Airline = stops_info['Airline'].values[0]

            # Prepare user input for prediction
            user_input_df = pd.DataFrame({
                'Year': [pd.to_datetime(depart_date).year],
                'Month': [pd.to_datetime(depart_date).month],
                'Day': [pd.to_datetime(depart_date).day],
                'Source': [depart_place],
                'Destination': [arrival_place],
            })

            encoded_user_input = pd.DataFrame(encoder.transform(user_input_df[['Source', 'Destination']]).toarray())
            encoded_user_input.columns = encoder.get_feature_names_out(['Source', 'Destination'])
            user_input_df = pd.concat([user_input_df.drop(['Source', 'Destination'], axis=1), encoded_user_input], axis=1)

            base_predicted_fare = selected_model.predict(user_input_df)[0]
            increase_percentage = 0.1  # Example increase for stops, etc.
            total_price = base_predicted_fare * num_persons * (1 + increase_percentage)

            if st.button('Predict'):
                st.success(
                    f"The predicted fare for {num_persons} person(s) on {Airline} from {depart_place} to {arrival_place} "
                    f"on {depart_date} with {num_stops} stop(s) is: Rs. {total_price:.2f}"
                )

# Reset button outside prediction block
if st.button('Reset'):
    st.session_state.depart_date = ''
    st.session_state.depart_place = ''
    st.session_state.arrival_place = ''
    st.session_state.num_persons = 1
    st.experimental_rerun()
