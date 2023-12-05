from flask import Flask,render_template,request, jsonify
import pickle
import config
import numpy as np
import pandas as pd
from utilities import read_extract_text_file,preprocess_data
app = Flask(__name__)

with open(config.model_path, "rb") as file:
    model = pickle.load(file)

with open(config.tfidf_path, "rb") as file1:
    tfidf = pickle.load(file1)

labels = ['Politics', 'Sport', 'Technology', 'Entertainment', 'Business']

@app.route("/")
def Home_Page():
    return render_template("index.html")

@app.route("/prediction", methods=['POST', 'GET'])
def prediction():
    if request.method=="POST":
        text_file = request.files['file']
        if text_file:
            #text_data = read_extract_text_file(text_file)
            text_data = text_file.read().decode('utf-8')
            clean_text = preprocess_data(text_data)
            tfidf_data = tfidf.transform([clean_text])

            y_pred = model.predict(tfidf_data.A)

            output = f"Congratulations!!! You Have Classify Given Document is Related To: {labels[y_pred[0]]}"

    return render_template("index.html", result=output)


if __name__=="__main__":
    app.run(debug=True, port=config.PORT, host=config.HOST)
