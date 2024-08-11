# app.py
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from models.symptom_data import X, y, symptom_data  # Updated import
import os
import json

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/healthguardian_model.h5')

# Load label encoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        symptom_vector = [0] * len(symptom_data['headache'])

        for symptom in symptoms:
            if symptom in symptom_data:
                symptom_vector = np.add(symptom_vector, symptom_data[symptom])

        symptom_vector = np.array(symptom_vector).reshape(1, -1)
        prediction_encoded = np.argmax(model.predict(symptom_vector), axis=1)
        prediction = le.inverse_transform(prediction_encoded)[0]

        # Save user history
        user_history = {
            'symptoms': symptoms,
            'prediction': prediction
        }
        with open('user_history.json', 'a') as f:
            json.dump(user_history, f)
            f.write('\n')

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

@app.route('/history')
def history():
    if os.path.exists('user_history.json'):
        with open('user_history.json', 'r') as f:
            history = [json.loads(line) for line in f]
    else:
        history = []

    return render_template('history.html', history=history)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback = request.form['feedback']
        with open('feedback.json', 'a') as f:
            json.dump({'feedback': feedback}, f)
            f.write('\n')
        return redirect(url_for('index'))
    
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
