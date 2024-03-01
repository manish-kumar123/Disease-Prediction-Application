import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load data from CSV file
df = pd.read_csv('patient_data.csv')

# Split the data into features and target
X = df[['Age', 'Blood Pressure']]
y = df['Disease']
imputer = SimpleImputer(strategy='mean')

# Create a pipeline with imputer and logistic regression
pipeline = make_pipeline(imputer)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict disease based on input
def predict_disease():
    age = int(entry_age.get())
    blood_pressure = int(entry_blood_pressure.get())
    # Make predictions
    prediction = model.predict([[age, blood_pressure]])
    if prediction[0] == 1:
        result = "Patient likely has the disease."
    else:
        result = "Patient likely doesn't have the disease."
    messagebox.showinfo("Prediction Result", result)

# Tkinter GUI
root = tk.Tk()
root.title("Disease Prediction")

label_age = tk.Label(root, text="Age:")
label_age.grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

label_blood_pressure = tk.Label(root, text="Blood Pressure:")
label_blood_pressure.grid(row=1, column=0)
entry_blood_pressure = tk.Entry(root)
entry_blood_pressure.grid(row=1, column=1)

button_predict = tk.Button(root, text="Predict", command=predict_disease)
button_predict.grid(row=2, column=0, columnspan=2)

root.mainloop()
