import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
import seaborn as sns


# Set page configuration
st.set_page_config(page_title="Rain Prediction App", layout="wide")

# Title and description
st.title("Rain Prediction Application")
st.write("This app predicts whether it will rain based on weather parameters.")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('rain.csv')
    return df

# Load the data
df = load_data()

# Prepare the data
X = df.drop(['Rain'], axis=1)
y = df['Rain']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
@st.cache_resource
def train_model():
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

model = train_model()

# Sidebar for user input
st.sidebar.header("Enter Weather Parameters")

def user_input_features():
    temperature = st.sidebar.slider('Temperature (°C)', float(df['Temperature'].min()), 
                                  float(df['Temperature'].max()), float(df['Temperature'].mean()))
    humidity = st.sidebar.slider('Humidity (%)', float(df['Humidity'].min()), 
                               float(df['Humidity'].max()), float(df['Humidity'].mean()))
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', float(df['Wind_Speed'].min()), 
                                 float(df['Wind_Speed'].max()), float(df['Wind_Speed'].mean()))
    cloud_cover = st.sidebar.slider('Cloud Cover (%)', float(df['Cloud_Cover'].min()), 
                                  float(df['Cloud_Cover'].max()), float(df['Cloud_Cover'].mean()))
    pressure = st.sidebar.slider('Pressure (hPa)', float(df['Pressure'].min()), 
                               float(df['Pressure'].max()), float(df['Pressure'].mean()))
    
    data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'Cloud_Cover': cloud_cover,
        'Pressure': pressure
    }
    return pd.DataFrame(data, index=[0])

# Main panel
st.header("Model Performance")
accuracy = model.score(X_test_scaled, y_test)
st.write(f"Model Accuracy: {accuracy:.2%}")

# Display training data
st.subheader("Sample Training Data")
st.write(df.head())

# User input section
st.header("Predict Rain")
user_data = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(user_data)

# Make prediction
user_data_scaled = scaler.transform(user_data)
prediction = model.predict(user_data_scaled)
prediction_proba = model.predict_proba(user_data_scaled)

# Display prediction
st.subheader('Prediction')
rain_prediction = prediction[0]
st.write('Prediction:', "Rain" if rain_prediction == "rain" else "No Rain")

st.subheader('Prediction Probability')
st.write(f"Probability of No Rain: {prediction_proba[0][0]:.2%}")
st.write(f"Probability of Rain: {prediction_proba[0][1]:.2%}")

# Visualization
st.header("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

st.bar_chart(feature_importance.set_index('Feature')) 
# Add interactive visualization section
st.header("Data Visualization")

# Correlation heatmap using plotly
import plotly.express as px
import matplotlib.pyplot as plt

# Allow user to select visualization type
viz_type = st.selectbox("Select Visualization", 
    ["Correlation Heatmap", "Feature Distribution", "Rain Distribution by Feature"])

if viz_type == "Correlation Heatmap":
    # Create a copy of dataframe and convert 'Rain' to numeric
    df_corr = df.copy()
    df_corr['Rain'] = df_corr['Rain'].map({'rain': 1, 'no rain': 0})
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig)

elif viz_type == "Feature Distribution":
    feature = st.selectbox("Select Feature", X.columns)
    fig = px.histogram(df, x=feature, color='Rain', marginal="box")
    st.plotly_chart(fig)

elif viz_type == "Rain Distribution by Feature":
    feature = st.selectbox("Select Feature", X.columns)
    fig = px.box(df, x='Rain', y=feature)
    st.plotly_chart(fig)

