import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('XGBmodel.pkl', 'rb'))
with open('scaling.pkl', 'rb') as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_apii', methods=['POST'])
def predict_api():
    try:
        # Get the data from the POST request
        data = request.json['data']
        
        # Convert data into numpy array and reshape it
        array_data = np.array(list(data.values())).reshape(1, -1)
        
        # Transform the data using the scaler
        scaled_data = scalar.transform(array_data)
        
        # Predict using the model
        output = model.predict(scaled_data)
        
        # Return the result as a JSON response
        return jsonify({'prediction': output[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        data = [float(x) for x in request.form.values()]
        
        # Convert data into numpy array and reshape it
        final_input = np.array(data).reshape(1, -1)
        
        # Transform the data using the scaler
        transformed_input = scalar.transform(final_input)
        
        # Predict using the model
        output = model.predict(transformed_input)[0]
        
        # Render the result in the template
        return render_template("home.html", prediction_text="The Credit Card Fraud Detection is: {}".format(output))
    
    except Exception as e:
        return render_template("home.html", prediction_text="An error occurred: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
