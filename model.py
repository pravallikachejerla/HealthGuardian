# models/model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from models.symptom_data import X, y

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Build the neural network model
model = Sequential([
    Dense(16, input_shape=(X.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=100, verbose=1)

# Save the model
model.save('models/healthguardian_model.h5')

print("Model training complete and saved as 'models/healthguardian_model.h5'.")
