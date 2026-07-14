from flask import Flask, render_template, request
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load model and scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Open prediction page
    if request.method == "GET":
        return render_template("index.html")

    try:
        # Read values from form
        cloud = float(request.form["cloud"])
        annual = float(request.form["annual"])
        janfeb = float(request.form["janfeb"])
        marmay = float(request.form["marmay"])
        junsep = float(request.form["junsep"])

        print("\n========== INPUT ==========")
        print(cloud, annual, janfeb, marmay, junsep)

        # Create dataframe
        data = pd.DataFrame(
            [[cloud, annual, janfeb, marmay, junsep]],
            columns=[
                "Cloud Cover",
                "ANNUAL",
                "Jan-Feb",
                "Mar-May",
                "Jun-Sep",
            ],
        )

        print("\nOriginal Data")
        print(data)

        # Scale data
        scaled = scaler.transform(data)

        print("\nScaled Data")
        print(scaled)

        print("\nBefore Prediction")

        prediction = model.predict(scaled)

        print("After Prediction")
        print(prediction)

        prediction = int(prediction[0])

        print("Final Prediction =", prediction)

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("no_chance.html")

    except Exception as e:

        print("\n========== ERROR ==========")
        traceback.print_exc()

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
        </head>
        <body style="font-family:Arial;padding:40px">
            <h1 style="color:red;">Prediction Failed</h1>
            <hr>
            <h3>Error Details</h3>
            <pre>{e}</pre>
        </body>
        </html>
        """


if __name__ == "__main__":
    app.run(debug=True)