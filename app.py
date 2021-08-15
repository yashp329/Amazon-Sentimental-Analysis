from os import name
import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
import pickle

app = Flask(__name__)
count_vectorizer = pickle.load(open('count_vectorizer3.pickle','rb'))
rndf_classifier = pickle.load(open('rndf_classifier4.pickle','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods = ["POST"])
def predict():
    reviews = str(request.form['reviews'])
    new_data = [reviews]
    new_vector = count_vectorizer.transform(new_data)
    pred = rndf_classifier.predict(new_vector)
    print(pred)
    return render_template("index.html", prediction_text = "The sentiment of the product review is {}".format(pred[0].upper()))


if __name__ == "__main__":
    app.run(debug = True)