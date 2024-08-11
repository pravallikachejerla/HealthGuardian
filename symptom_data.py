import numpy as np
import pandas as pd

# Example symptom and condition data
symptom_data = {
    'headache': [1, 0, 0, 0, 0, 0, 0, 0],
    'fever': [0, 1, 0, 0, 0, 0, 0, 0],
    'nausea': [0, 0, 1, 0, 0, 0, 0, 0],
    'cough': [0, 0, 0, 1, 0, 0, 0, 0],
    'fatigue': [0, 0, 0, 0, 1, 0, 0, 0],
    'sore throat': [0, 0, 0, 0, 0, 1, 0, 0],
    'muscle pain': [0, 0, 0, 0, 0, 0, 1, 0],
    'shortness of breath': [0, 0, 0, 0, 0, 0, 0, 1]
}

conditions = [
    'Migraine', 
    'Flu', 
    'Food Poisoning', 
    'Common Cold', 
    'Fatigue Syndrome', 
    'Tonsillitis',
    'Muscle Strain',
    'Bronchitis'
]

# Convert data to DataFrame
df = pd.DataFrame(symptom_data, index=conditions)

# Prepare data for the model
X = df.values
y = np.array(conditions)
