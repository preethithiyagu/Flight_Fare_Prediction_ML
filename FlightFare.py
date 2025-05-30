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

# User Input
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

# Convert Date_of_Journey to datetime
model_data['Date_of_Journey'] = pd.to_datetime(model_data['Date_of_Journey'], format='%d/%m/%Y')
model_data['Year'] = model_data['Date_of_Journey'].dt.year
model_data['Month'] = model_data['Date_of_Journey'].dt.month
model_data['Day'] = model_data['Date_of_Journey'].dt.day
model_data = model_data.drop(['Date_of_Journey'], axis=1)

# Convert categorical variables into numerical representations
encoder = OneHotEncoder(handle_unknown='ignore')

# Check for missing values in categorical columns
if model_data[['Source', 'Destination']].isnull().any().any():
    st.write("Error: Missing values found in categorical columns. Please handle missing values first.")
else:
    try:
        encoded_cols = encoder.fit_transform(model_data[['Source', 'Destination']])
        column_names = encoder.get_feature_names_out(['Source', 'Destination'])

        if not set(['Source', 'Destination']).issubset(model_data.columns):
            st.write("Error: Columns 'Source' and 'Destination' not found in the DataFrame.")
        else:
            model_data_encoded = pd.DataFrame(encoded_cols.toarray(), columns=column_names)

            # Concatenate encoded columns with original DataFrame
            model_data_encoded = pd.concat([model_data.drop(['Source', 'Destination'], axis=1).reset_index(drop=True),
                                            model_data_encoded.reset_index(drop=True)], axis=1)
    except ValueError as e:
        st.write(f"Error: {e}")

# Check available sources and destinations
available_sources = set(data['Source'].unique())
available_destinations = set(data['Destination'].unique())

if depart_place and arrival_place:
    if (depart_place not in available_sources) or (arrival_place not in available_destinations):
        st.write("No flights available for the provided source and/or destination.")
    else:
        direct_flight_check = len(data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)])
        if direct_flight_check == 0:
            st.write("No direct flights available for the specified route.")

# Prepare data for training
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
    score = model.score(X_test, y_test)
    # Optional: Display training scores in Streamlit or console
    # st.write(f"{name} - Test R2 Score: {score:.4f}")

selected_model = models['Random Forest Regressor']

# Prediction logic when user clicks Predict
if st.button('Predict'):
    # Validate inputs
    if not depart_date or not validate_date_format(depart_date):
        st.write("Please enter a valid departure date in YYYY-MM-DD format.")
    elif not depart_place or not arrival_place:
        st.write("Please enter both departure and arrival places.")
    elif num_persons < 1:
        st.write("Number of persons must be at least 1.")
    else:
        stops_info = data[(data['Source'] == depart_place) & (data['Destination'] == arrival_place)]

        if stops_info.empty:
            st.write("No flights found for the selected route.")
        else:
            num_stops = stops_info['Total_Stops'].values[0]
            Airline = stops_info['Airline'].values[0]

            # Prepare input for prediction
            user_input_df = pd.DataFrame({
                'Year': [pd.to_datetime(depart_date).year],
                'Month': [pd.to_datetime(depart_date).month],
                'Day': [pd.to_datetime(depart_date).day],
                'Source': [depart_place],
                'Destination': [arrival_place],
            })

            encoded_user_input = pd.DataFrame(encoder.transform(user_input_df[['Source', 'Destination']]).toarray())
            encoded_user_input.columns = encoder.get_feature_names_out(['Source', 'Destination'])
            user_input_df = pd.concat([user_input_df.reset_index(drop=True), encoded_user_input.reset_index(drop=True)], axis=1)
            user_input_df = user_input_df.drop(['Source', 'Destination'], axis=1)

            base_predicted_fare = selected_model.predict(user_input_df)
            increase_percentage = 0.1
            total_price = base_predicted_fare[0] * num_persons * (1 + increase_percentage)

            st.write(f"The predicted fare for {num_persons} persons on {Airline} from {depart_place} to {arrival_place} "
                     f"on {depart_date} with {num_stops} stop(s) is: Rs. {total_price:.2f}")

# Optional Reset button
if st.button('Reset'):
    depart_date = ''
    depart_place = ''
    arrival_place = ''
    num_persons = 1
    st.experimental_rerun()
