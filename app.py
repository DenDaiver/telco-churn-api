from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('churn_model.pkl')
processor = joblib.load('churn_processor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    processed = processor.transform(df)
    prediction = model.predict(processed)[0]
    result = "Churn" if prediction == 1 else "No Churn"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
