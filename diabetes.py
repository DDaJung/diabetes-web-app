# diabetes.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from tensorflow import keras

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# FlaskForm definition
class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Home route
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # Extract input data
        X_test = np.array([[
            float(form.preg.data),
            float(form.glucose.data),
            float(form.blood.data),
            float(form.skin.data),
            float(form.insulin.data),
            float(form.bmi.data),
            float(form.dpf.data),
            float(form.age.data)]])

        # Load diabetes dataset
        data = pd.read_csv('./diabetes.csv', sep=',')
        X = data.values[:, 0:8]
        y = data.values[:, 8]

        # Normalize data using MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_test = scaler.transform(X_test)

        # Load trained model
        model = keras.models.load_model('pima_model.h5')


        # Make prediction
        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res, 2)
        res = float(np.round(res * 100))  # percent

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
