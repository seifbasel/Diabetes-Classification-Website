from flask import Flask, request, render_template
import predictDiabetes

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict() :
    button = request.form.get('predict_button')
    if button == "knn_predict" :
        prediction = predictDiabetes.predict_diabetes_KNN(request.form['cholesterol'], request.form['glucose'], request.form['hdl_chol'],
                                                    request.form['age'], request.form['weight'], request.form['systolic_bp'], request.form['diastolic_bp'])
    else : 
        prediction = predictDiabetes.predict_diabetes_SVM(request.form['cholesterol'], request.form['glucose'], request.form['hdl_chol'],
                                                    request.form['age'], request.form['weight'], request.form['systolic_bp'], request.form['diastolic_bp'])
    if prediction == 0 :
        return render_template('not_diabetic.html')
    else :
        return render_template('diabetic.html')
    
if __name__ == '__main__' :
    app.run(debug=True)