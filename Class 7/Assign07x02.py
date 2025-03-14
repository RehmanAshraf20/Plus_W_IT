import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset (Replace 'salary_data.csv' with an actual dataset)
df = pd.read_csv('salary_data.csv')
# Preprocess data
df.dropna(inplace=True)
# Identify categorical columns
categorical_columns = ['degree', 'job_role', 'location']
existing_categorical_columns = [col for col in categorical_columns if col in
df.columns]
# Apply one-hot encoding only if columns exist
if existing_categorical_columns:
    df = pd.get_dummies(df, columns=existing_categorical_columns,drop_first=True)
# Define features and target variable
if 'Salary' in df.columns:
    X = df.drop(columns=['Salary'])
    y = df['Salary']
# Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
# Predictions
    y_pred = model.predict(X_test)
# Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse}, R-squared: {r2}')
# Example prediction
    sample_input = X_test.iloc[[0]] # Keep feature names
    predicted_salary = model.predict(sample_input)
    print(f'Predicted Salary for sample input: {predicted_salary[0]}')
else:
    print("Error: The 'Salary' column is missing from the dataset.")