import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt


# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('D:\dataaaaaaa\F1- Mexico pred-weather\ocean_mexican_gp_data_challenge_datasets\Weather_Grid_Stint\Weather_Grid_Model.csv')
    return data

data = load_data()

# Function to convert lap time strings to seconds
def convert_lap_time_to_seconds(lap_time):
    if lap_time == '0':
        return np.nan
    parts = lap_time.split('.')
    if len(parts) == 3:
        minutes, seconds, milliseconds = map(int, parts)
        return minutes * 60 + seconds + milliseconds / 1000
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    return float('nan')

# Apply the conversion to the LapTime column
data['LapTime'] = data['LapTime'].apply(convert_lap_time_to_seconds)

# Exclude rows where GridPosition or LapTime is 0 or NaN
data = data[(data['GridPosition'] != 0) & (data['LapTime'].notna())].copy()

# Preprocess the data
le_driver = LabelEncoder()
le_compound = LabelEncoder()
data['Driver'] = le_driver.fit_transform(data['Driver'])
data['Compound'] = le_compound.fit_transform(data['Compound'])

# Calculate additional features
data['LapsInStint'] = data.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapNumber'].transform('count')
data['AverageLapTime'] = data.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapTime'].transform('mean')
data['TireDegradation'] = data.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapTime'].transform(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
data['TotalRaceLaps'] = data.groupby(['Year', 'EventName'])['LapNumber'].transform('max')

# Prepare features and target
feature_columns = ['Driver', 'GridPosition', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'Stint', 'TotalRaceLaps']
X = data[feature_columns]
y_laps_stint = data['LapsInStint']
y_avg_lap_time = data['AverageLapTime']
y_compound = data['Compound']
y_tire_degradation = data['TireDegradation']
y_num_stints = data.groupby(['Year', 'EventName', 'Driver'])['Stint'].transform('max')

# Split the data for each target separately
X_train, X_test, y_laps_stint_train, y_laps_stint_test = train_test_split(X, y_laps_stint, test_size=0.2, random_state=42)
X_train, X_test, y_avg_lap_time_train, y_avg_lap_time_test = train_test_split(X, y_avg_lap_time, test_size=0.2, random_state=42)
X_train, X_test, y_compound_train, y_compound_test = train_test_split(X, y_compound, test_size=0.2, random_state=42)
X_train, X_test, y_tire_degradation_train, y_tire_degradation_test = train_test_split(X, y_tire_degradation, test_size=0.2, random_state=42)
X_train, X_test, y_num_stints_train, y_num_stints_test = train_test_split(X, y_num_stints, test_size=0.2, random_state=42)

# Train the models
@st.cache_resource
def train_models():
    model_laps_stint = RandomForestRegressor(n_estimators=100, random_state=42)
    model_laps_stint.fit(X_train, y_laps_stint_train)

    model_avg_lap_time = RandomForestRegressor(n_estimators=100, random_state=42)
    model_avg_lap_time.fit(X_train, y_avg_lap_time_train)

    model_compound = RandomForestClassifier(n_estimators=100, random_state=42)
    model_compound.fit(X_train, y_compound_train)

    model_tire_degradation = RandomForestRegressor(n_estimators=100, random_state=42)
    model_tire_degradation.fit(X_train, y_tire_degradation_train)

    model_num_stints = RandomForestRegressor(n_estimators=100, random_state=42)
    model_num_stints.fit(X_train, y_num_stints_train)

    return model_laps_stint, model_avg_lap_time, model_compound, model_tire_degradation, model_num_stints

model_laps_stint, model_avg_lap_time, model_compound, model_tire_degradation, model_num_stints = train_models()

# Make predictions
y_pred_laps_stint = model_laps_stint.predict(X_test)
y_pred_avg_lap_time = model_avg_lap_time.predict(X_test)
y_pred_compound = model_compound.predict(X_test)
y_pred_tire_degradation = model_tire_degradation.predict(X_test)
y_pred_num_stints = model_num_stints.predict(X_test)

# Calculate evaluation metrics
metrics = {}

# Stint Model
metrics['Stint Model'] = {
    'MSE': mean_squared_error(y_laps_stint_test, y_pred_laps_stint),
    'R2 Score': model_laps_stint.score(X_test, y_laps_stint_test)
}

# Average Lap Time Model
metrics['Average Lap Time Model'] = {
    'MSE': mean_squared_error(y_avg_lap_time_test, y_pred_avg_lap_time),
    'R2 Score': model_avg_lap_time.score(X_test, y_avg_lap_time_test)
}

# Compound Model (Classification)
metrics['Compound Model'] = {
    'Accuracy': accuracy_score(y_compound_test, y_pred_compound)
}

# Tire Degradation Model
metrics['Tire Degradation Model'] = {
    'MSE': mean_squared_error(y_tire_degradation_test, y_pred_tire_degradation),
    'R2 Score': model_tire_degradation.score(X_test, y_tire_degradation_test)
}

# Number of Stints Model
metrics['Number of Stints Model'] = {
    'MSE': mean_squared_error(y_num_stints_test, y_pred_num_stints),
    'R2 Score': model_num_stints.score(X_test, y_num_stints_test)
}

# Output metrics
for model, scores in metrics.items():
    print(f"{model}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    print()


# Function to parse input ranges
def parse_input_range(input_str):
    try:
        if '-' in input_str:
            low, high = map(float, input_str.split('-'))
            return np.linspace(low, high, num=5)
        else:
            return [float(input_str)]
    except ValueError:
        st.error(f"Invalid input format: {input_str}. Please use a single number or 'min-max' format.")
        return []

# Function to predict strategy
def predict_strategy(drivers, grid_positions, air_temp, humidity, pressure, rainfall, track_temp, wind_direction, wind_speed, total_race_laps):
    try:
        rainfall_numeric = 1 if rainfall else 0
        
        grid_positions = [int(pos.strip()) for pos in grid_positions.split(',')]
        
        if len(drivers) != len(grid_positions):
            return "Error: Number of drivers must match the number of grid positions."

        total_race_laps = int(total_race_laps)

        air_temp_range = parse_input_range(air_temp)
        humidity_range = parse_input_range(humidity)
        pressure_range = parse_input_range(pressure)
        track_temp_range = parse_input_range(track_temp)
        wind_direction_range = parse_input_range(wind_direction)
        wind_speed_range = parse_input_range(wind_speed)

        results = []

        for driver, grid_position in zip(drivers, grid_positions):
            driver = driver.strip()

            input_combinations = product(
                air_temp_range, humidity_range, pressure_range,
                track_temp_range, wind_direction_range, wind_speed_range
            )

            all_predictions = []

            for air_temp, humidity, pressure, track_temp, wind_direction, wind_speed in input_combinations:
                input_data = pd.DataFrame([[
                    le_driver.transform([driver])[0], grid_position, air_temp, humidity, pressure,
                    rainfall_numeric, track_temp, wind_direction, wind_speed, 1, total_race_laps
                ]], columns=feature_columns)

                predicted_num_stints = max(2, min(5, int(round(model_num_stints.predict(input_data)[0]))))

                stints = []
                total_laps = 0
                last_avg_lap_time = None

                for stint in range(1, predicted_num_stints + 1):
                    input_data['Stint'] = stint
                    
                    predicted_laps = max(1, min(35, int(round(model_laps_stint.predict(input_data)[0]))))
                    predicted_avg_lap_time = max(60, min(120, model_avg_lap_time.predict(input_data)[0]))
                    predicted_compound = le_compound.inverse_transform(model_compound.predict(input_data))[0]
                    predicted_degradation = max(0, model_tire_degradation.predict(input_data)[0])

                    if stint == predicted_num_stints:
                        predicted_laps = max(1, total_race_laps - total_laps)

                    total_laps += predicted_laps

                    if last_avg_lap_time is not None:
                        relative_degradation = predicted_avg_lap_time - last_avg_lap_time
                    else:
                        relative_degradation = 0

                    stints.append((stint, predicted_laps, predicted_avg_lap_time, predicted_compound, relative_degradation))
                    last_avg_lap_time = predicted_avg_lap_time

                    if total_laps >= total_race_laps:
                        break

                all_predictions.append(stints)
            
            avg_predictions = []
            for stint_index in range(max(len(stints) for stints in all_predictions)):
                avg_stint = [stint_index + 1]
                for metric in range(1, 5):
                    values = [stints[stint_index][metric] for stints in all_predictions if stint_index < len(stints)]
                    if metric == 3:
                        avg_stint.append(max(set(values), key=values.count))
                    else:
                        avg_stint.append(sum(values) / len(values))
                avg_predictions.append(tuple(avg_stint))

            results.append({
                "driver": driver,
                "grid_position": grid_position,
                "stints": avg_predictions,
                "total_laps": total_race_laps
            })

        return results

    except Exception as e:
        return f"Error in predict_strategy: {str(e)}"

# Streamlit app
st.title("F1 Tire Strategy Predictor")
st.write("Predict tire strategies for F1 races based on weather and grid information. You can enter single values or ranges for weather conditions.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    drivers = st.multiselect("Select Drivers", options=le_driver.classes_.tolist())
    grid_positions = st.text_input("Grid Positions (comma-separated, e.g., 1, 2, 3)")
    total_race_laps = st.number_input("Total Race Laps", min_value=1, max_value=100, value=50)

with col2:
    air_temp = st.text_input("Air Temperature (°C) - single value or range", "18-25")
    humidity = st.text_input("Humidity (%) - single value or range", "60-80")
    pressure = st.text_input("Pressure (mbar) - single value or range", "1010-1020")
    rainfall = st.checkbox("Rainfall")
    track_temp = st.text_input("Track Temperature (°C) - single value or range", "30-40")
    wind_direction = st.text_input("Wind Direction (°) - single value or range", "170-190")
    wind_speed = st.text_input("Wind Speed (km/h) - single value or range", "10-20")

if st.button("Predict Tire Strategy"):
    if not drivers or not grid_positions:
        st.error("Please select at least one driver and provide grid positions.")
    else:
        results = predict_strategy(
            drivers, grid_positions, air_temp, humidity, pressure, rainfall,
            track_temp, wind_direction, wind_speed, total_race_laps
        )

        if isinstance(results, str):
            st.error(results)
        else:
            for result in results:
                st.subheader(f"Driver: {result['driver']}, Grid Position: {result['grid_position']}")
                
                # Create a DataFrame for the stint data
                stint_data = pd.DataFrame(result['stints'], columns=['Stint', 'Laps', 'Avg Laptime', 'Compound', 'Relative Pace'])
                
                # Display stint information
                st.write(stint_data)

                # Create a stacked bar chart for stint composition
                fig, ax = plt.subplots(figsize=(10, 6))
                bottom = np.zeros(len(stint_data))
                
                for compound in stint_data['Compound'].unique():
                    mask = stint_data['Compound'] == compound
                    ax.bar(stint_data['Stint'], stint_data['Laps'][mask], bottom=bottom[mask], label=compound)
                    bottom += stint_data['Laps'] * mask
                
                ax.set_xlabel('Stint')
                ax.set_ylabel('Laps')
                ax.set_title(f'Stint Composition for {result["driver"]}')
                ax.legend(title='Compound')
                st.pyplot(fig)

                # Create a line plot for average lap times
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=stint_data, x='Stint', y='Avg Laptime', marker='o')
                ax.set_xlabel('Stint')
                ax.set_ylabel('Average Lap Time (seconds)')
                ax.set_title(f'Average Lap Times per Stint for {result["driver"]}')
                st.pyplot(fig)

                # Create a heatmap for relative pace
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(stint_data[['Stint', 'Relative Pace']].set_index('Stint').T, annot=True, cmap='RdYlGn_r', center=0)
                ax.set_title(f'Relative Pace per Stint for {result["driver"]}')
                st.pyplot(fig)

                st.write("---")

#if __name__ == "__main__":
 #   st.run()
