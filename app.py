from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Ensure correct path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "titanic_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        sex = 1 if request.form["sex"].lower() == "female" else 0
        age = float(request.form["age"])
        pclass = int(request.form["pclass"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])

        # Create input for model
        input_features = [[pclass, sex, age, sibsp, parch, fare]]
        prediction = model.predict(input_features)[0]

        result = "Survived" if prediction == 1 else "Did not survive"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    # Run on Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
