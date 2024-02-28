import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

model = pickle.load(open('hiring_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route("/predict", methods=["POST"])
def predict():
    input_values = [int(x) for x in request.form.values()]
    final_input = [np.array(input_values)]
    prediction = model.predict(final_input)    
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text="Employee Salary should be $ {}".format(output))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)