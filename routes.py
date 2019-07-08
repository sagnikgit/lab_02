from flask import Flask, render_template
import pandas as pandas
import sklearn
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/")
def index():
	model = joblib.load('regressor.pkl')
	prediction = int(model.predict([[5, 100, 5000, 52]]).round(1))
	return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
	app.run(debug=True)