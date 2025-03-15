from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and selected feature names
model = joblib.load("best_house_price_model.pkl")
selected_features = joblib.load("selected_features.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request
        data = request.get_json()
        print("Received JSON from Power BI:", data)  # Debugging step

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure correct features are present
        missing_cols = [col for col in selected_features if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # Use only selected features
        input_df = input_df[selected_features]

        # Predict price
        predicted_price = model.predict(input_df)[0]

        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
