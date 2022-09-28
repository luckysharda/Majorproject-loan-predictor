from flask import Flask, request, jsonify, render_template

import numpy as np
import pickle


app = Flask(__name__)


model_RF = pickle.load(open('Major_RF.pkl', 'rb'))
model_KNN = pickle.load(open('Major_KNN.pkl', 'rb'))
model_K_SVM = pickle.load(open('Major_SVM_linear.pkl', 'rb'))
model_DT = pickle.load(open('Major_DT.pkl', 'rb'))
model_NB = pickle.load(open('Major_NB.pkl', 'rb'))


@app.route('/')
def home():

    return render_template("index.html")
# ------------------------------About us-------------------------------------------


@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')


@app.route('/predict', methods=['GET'])
def predict():

    Gender = float(request.args.get('Gender'))
    Married = float(request.args.get('Married'))
    Education = float(request.args.get('Education'))
    Self_Employed = float(request.args.get('Self_Employed'))
    ApplicantIncome = float(request.args.get('ApplicantIncome'))
    LoanAmount = float(request.args.get('LoanAmount'))
    Loan_Amount_Term = float(request.args.get('Loan_Amount_Term'))
    Property_Area = float(request.args.get('Property_Area'))

    Model = (request.args.get('Model'))

    if Model == "Random Forest Classifier":
        prediction = model_RF.predict(
            [[Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Property_Area]])

    elif Model == "Decision Tree Classifier":
        prediction = model_DT.predict(
            [[Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Property_Area]])

    elif Model == "KNN Classifier":
        prediction = model_KNN.predict(
            [[Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Property_Area]])

    elif Model == "SVM Classifier":
        prediction = model_K_SVM.predict(
            [[Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Property_Area]])

    else:
        prediction = model_NB.predict(
            [[Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Property_Area]])

    if prediction == [1]:
        return render_template('index.html', prediction_text='You are eligible for loan', extra_text=" -- Prediction by " + Model)

    else:
        return render_template('index.html', prediction_text='Sorry,You are not eligible for loan', extra_text=" -- Prediction by " + Model)


# app.run()
# if __name__ == "__main__":
#     app.run(debug=True)
