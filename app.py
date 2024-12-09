from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('Régression_logistique_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Récupérer les données de l'utilisateur
            age = float(request.form['age'])
            hypertension = 1 if request.form['hypertension'] == 'Oui' else 0
            heart_disease = 1 if request.form['heart_disease'] == 'Oui' else 0
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            # Mettre les données sous forme de tableau
            features = np.array([
                age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level
            ]).reshape(1, -1)

            # Normaliser les données
            features_scaled = scaler.transform(features)

            # Faire la prédiction
            prediction = model.predict(features_scaled)

            # Afficher le résultat
            if prediction == 1:
                prediction_text = "Le patient est diabétique."
            else:
                prediction_text = "Le patient n'est pas diabétique."

            return render_template('index.html', prediction_text=prediction_text)
        
        except Exception as e:
            return render_template('index.html', prediction_text=f"Erreur dans la prédiction : {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
