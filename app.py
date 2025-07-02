from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('churn_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return '✅ Churn Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON data from Salesforce or frontend
        data = request.get_json()
        total_orders = data['TotalOrders']
        last_order_days = data['LastOrderDaysAgo']
        
        # Predict
        df = pd.DataFrame([[total_orders, last_order_days]], columns=['TotalOrders', 'LastOrderDaysAgo'])
        prediction = model.predict(df)[0]
        
        return jsonify({'churn_prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
