from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_scalar=pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/humu",methods=["GET","POST"])
def predict_datapoint():
    if request.method=='POST':
        temperature=float(request.form.get("temperature"))
        RH=float(request.form.get("RH"))
        WS=float(request.form.get("WS"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        classes=float(request.form.get("classes"))
        region=float(request.form.get("region"))

        new_data_scaled=standard_scalar.transform([[temperature,RH,WS,Rain,FFMC,DMC,ISI,classes,region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",port="5005",debug=True)