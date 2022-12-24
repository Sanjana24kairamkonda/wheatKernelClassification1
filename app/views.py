from app import app
from flask import Flask,request,jsonify,render_template
import pickle
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods= ["POST"])
def predict():
    float_features= [float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction= model.predict(features)

    return render_template("index.html", prediction_text = "wheat classification: {}".format(prediction))
