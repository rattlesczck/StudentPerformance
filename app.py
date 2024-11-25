import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset and preprocess the data
df = pd.read_csv("C:\\Users\\ivan_\\OneDrive\\Desktop\\Studentperformance\\StudentData.csv")
df['Gender'].replace({'Female': 'F', 'Male': 'M'}, inplace=True)
df['Roll'].fillna(0, inplace=True)
df['Roll no.'].fillna(0, inplace=True)
df['Gender'].fillna('Unknown', inplace=True) 
df['Roll_no'] = (df['Roll'] + df['Roll no.']).astype(int)
df.drop(columns=['Roll', 'Roll no.'], inplace=True)
df['1st'].fillna(df['1st'].mean(), inplace=True)
df['2nd'].fillna(df['2nd'].mean(), inplace=True)
df['3rd'].fillna(df['3rd'].mean(), inplace=True)
df['4th'].fillna(df['4th'].mean(), inplace=True)
df['5th'].fillna(df['5th'].mean(), inplace=True)

# Features and target
X = df[['1st', '2nd', '3rd', '4th']]  # Features
y = df['5th']  # Target (5th semester score)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a model (Random Forest as an example)
model_rf = RandomForestRegressor(n_estimators=200, random_state=21)
model_rf.fit(X_scaled, y)

# Save the model using joblib
import joblib
joblib.dump(model_rf, 'model_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json(force=True)
    
    # Extract input features
    features = np.array([data['1st'], data['2nd'], data['3rd'], data['4th']]).reshape(1, -1)
    
    # Scale the input data
    scaled_features = scaler.transform(features)
    
    # Predict the future marks using the model
    prediction = model_rf.predict(scaled_features)
    
    # Return the prediction as a JSON response
    return jsonify({'Predicted_5th_Mark': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
