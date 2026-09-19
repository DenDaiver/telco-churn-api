## Create the Flask application itself; __name__ is a built-in Python
# variable Flask uses internally to locate files relative to this script
app = Flask(__name__)
run_with_ngrok(app) # Start ngrok when app is run

# Load the trained, tuned Logistic Regression model saved earlier
model = joblib.load('churn_model.pkl')
# Load the saved preprocessor (scaler + one-hot encoder) so we can
# transform new raw customer data the exact same way training data was transformed
processor = joblib.load('churn_processor.pkl')

# Define a route for the homepage ('/') - just a simple check to confirm
# the API is running when someone visits the base URL
@app.route('/')
  # Message shown when visiting the homepage
def home():
    return "Telco Churn Prediction API is running."

# Define the actual prediction route ('/predict'), which only accepts
# POST requests (i.e., data being sent TO this route, not just visited)
@app.route('/predict', methods=['POST'])
def predict():
   #Grab the incoming customer data, sent as JSON
    data = request.get_json()
  # Wrap that data into a 1-row pandas DataFrame, since the
 # preprocessor expects a DataFrame, not a raw dictionary
    df = pd.DataFrame([data])
    # Run the new customer's raw data through the saved preprocessing
    # rules (scaling numeric columns, one-hot encoding categorical ones)
    processed = processor.transform(df)
        # Use the trained model to predict churn (0 or 1) for this customer;
    # [0] grabs the single prediction, since we're only predicting one row
    prediction = model.predict(processed)[0]

    # Convert the raw 0/1 prediction into a human-readable label
    result = "Churn" if prediction == 1 else "No Churn"
        # Send the result back as a proper JSON response
    return jsonify({"prediction": result})

    app.run(host='0.0.0.0', port=5000)
