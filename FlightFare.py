import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

# Set Streamlit page config
st.set_page_config(page_title="Flight Fare Prediction", page_icon="🛫")
st.title("Flight Fare Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("a1_FlightFare_Dataset.csv")

data = load_data()

# --- Initialize session state variables ---
if 'depart_date' not in st.session_state:
    st.session_state.depart_date = pd.to_datetime('2023-01-01')

if 'depart_place' not in st.session_state:
    st.session_state.depart_place = ''

if 'arrival_place' not in st.session_state:
    st.session_state.arrival_place = ''

if 'num_persons' not in st.session_state:
    st.session_state.num_persons = 1

# --- Reset Button ---
if st.button('Reset'):
    st.session_state.depart_date = pd.to_datetime('2023-01-01')
    st.session_state.depart_place = ''
    st.session_state.arrival_place = ''
    st.session_state.num_persons = 1
    st.experimental_rerun()

# --- User Inputs ---
depart_date = st.date_input("Enter departure date:", value=st.session_state.depart_date)
depart_place = st.text_input("Enter departure place:", value=st.session_state.depart_place)
arrival_place = st.text_input("Enter arrival place:", value=st.session_state.arrival_place)
num_persons = st.number_input("Enter number of persons:", min_value=1, step=1, value=st.session_state.num_persons)

# --- Update session state ---
st.session_state.depart_date = depart_date
st.session_state.depart_place = depart_place
st.session_state.arrival_place = arrival_place
st.session_state.num_persons = num_persons

# Prepare dataset
selected_features = ['Date_of_Journey', 'Source', 'Destination', 'Price']
model_data = data[selected_features].copy()
model_data['Date_of_Journey'] = pd.to_datetime(model_data['Date_of_Journey'], format='%d/%m/%Y')
model_data['Year'] = model_data['Date_of_Journey'].dt.year
model_data['Month'] = model_data['Date_of_Journey'].dt.month
model_data['Day'] = model_data['Date_of_Journey'].dt.day
model_data.drop(['Date_of_Journey'], axis=1, inplace=True)

# Encode categorical features
encoder = OneHotEncoder()
encoded_cols = encoder.fit_transform(model_data[['Source', 'Destination']])
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(['Source', 'Destination']))
model_data_encoded = pd.concat([model_data.drop(['Source', 'Destination'], axis=1).reset_index(drop=True), encoded_df], axis=1)

# Available options
available_sources = set(data['Source'].unique())
available_destinations = set(data['Destination'].unique())

# --- Prediction Logic ---
if depart_place and arrival_place and num_persons >= 1:
    if depart_place not in available_sources or arrival_place not in available_destinations:
        st.warning("No flights available for the provided source and/or destination.")
    else:
        matching_flights = data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)]
        if matching_flights.empty:
            st.warning("No direct flights available for the specified route.")
        else:
            # Train models
            X = model_data_encoded.drop('Price', axis=1)
            y = model_data_encoded['Price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
            }

            for model in models.values():
                model.fit(X_train, y_train)

            selected_model = models['Random Forest Regressor']

            # Prepare user input
            user_input_df = pd.DataFrame({
                'Year': [depart_date.year],
                'Month': [depart_date.month],
                'Day': [depart_date.day],
                'Source': [depart_place],
                'Destination': [arrival_place]
            })

            user_encoded = encoder.transform(user_input_df[['Source', 'Destination']])
            user_encoded_df = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['Source', 'Destination']))
            user_input_final = pd.concat([user_input_df.drop(['Source', 'Destination'], axis=1), user_encoded_df], axis=1)

            # Predict
            base_price = selected_model.predict(user_input_final)[0]
            increase_percentage = 0.1  # Increase for assumed stops/services
            total_price = base_price * num_persons * (1 + increase_percentage)

            airline = matching_flights.iloc[0]['Airline']
            stops = matching_flights.iloc[0]['Total_Stops']

            if st.button('Predict'):
                st.success(
                    f"Predicted fare for {num_persons} person(s) on {airline} "
                    f"from {depart_place} to {arrival_place} on {depart_date.strftime('%Y-%m-%d')} "
                    f"with {stops} stop(s): ₹{total_price:.2f}"
                )
