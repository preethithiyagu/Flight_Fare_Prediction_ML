import pandas as pd
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

# User inputs
depart_date = input("Enter departure date (YYYY-MM-DD): ")
depart_place = input("Enter departure place: ")
arrival_place = input("Enter arrival place: ")
num_persons = int(input("Enter number of persons: "))

while not validate_date_format(depart_date):
    print("Invalid date format! Please use the format 'YYYY-MM-DD'.")
    depart_date = input("Enter departure date (YYYY-MM-DD): ")
    
# Feature selection for demonstration
selected_features = ['Date_of_Journey', 'Source', 'Destination', 'Price']
model_data = data[selected_features].copy()

model_data['Date_of_Journey'] = pd.to_datetime(model_data['Date_of_Journey'], format='%d/%m/%Y')
model_data.loc[:, 'Year'] = model_data['Date_of_Journey'].dt.year
model_data.loc[:, 'Month'] = model_data['Date_of_Journey'].dt.month
model_data.loc[:, 'Day'] = model_data['Date_of_Journey'].dt.day
model_data = model_data.drop(['Date_of_Journey'], axis=1).copy()

# Convert categorical variables into numerical representations
encoder = OneHotEncoder(sparse=False)
encoded_cols = pd.DataFrame(encoder.fit_transform(model_data[['Source', 'Destination']]))
encoded_cols.columns = encoder.get_feature_names_out(['Source', 'Destination'])  # Updated line
model_data = pd.concat([model_data, encoded_cols], axis=1).drop(['Source', 'Destination'], axis=1)

# Check available sources and destinations
available_sources = set(model_data.columns[model_data.columns.str.startswith('Source_')])
available_destinations = set(model_data.columns[model_data.columns.str.startswith('Destination_')])

if (f'Source_{depart_place}' not in available_sources) or (f'Destination_{arrival_place}' not in available_destinations):
    print("No flights available for the provided source and/or destination.")
else:
    X = model_data.drop('Price', axis=1)
    y = model_data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} - Test R2 Score: {score}")

    selected_model = models['Gradient Boosting Regressor']

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
        Airline = stops_info['Airline'].values[0]  # Assuming 'Flight_Name' is the column containing flight names

        print(f"The predicted fare for {num_persons} persons on {Airline} from {depart_place} to {arrival_place} on {depart_date} with {num_stops} is: Rs. {total_price}")
        
    else:
        print("No direct flights available for the specified route.")      
