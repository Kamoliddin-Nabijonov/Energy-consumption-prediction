import pickle
from flask import Flask
from flask import request
from flask import jsonify


dict_vectorizer_file = './models/dict_vectorizer.bin'
linear_regression_model_file = './models/linear_regression_model.bin'
svr_model_file = './models/svr_model.bin'
random_forest_regressor_model_file = './models/random_forest_regressor_model.bin'
gradient_boosting_regressor_model_file = './models/gradient_boosting_regressor_model.bin'

with open(dict_vectorizer_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(linear_regression_model_file, 'rb') as f_in:
    linear_regression = pickle.load(f_in)

with open(svr_model_file, 'rb') as f_in:
    svr = pickle.load(f_in)

with open(random_forest_regressor_model_file, 'rb') as f_in:
    random_forest = pickle.load(f_in)

with open(gradient_boosting_regressor_model_file, 'rb') as f_in:
    gradient_boosting = pickle.load(f_in)


app = Flask('energy')

@app.route('/forecast_demand', methods=['POST'])
def predict():

    request_data = request.get_json()
    X = dv.transform([request_data])

    prediction_linear_regression = linear_regression.predict(X)
    prediction_svr = svr.predict(X)
    prediction_random_forest_regressor = random_forest.predict(X)
    prediction_gradient_boosting_regressor = gradient_boosting.predict(X)
 
    result = {
        'prediction_linear_regression': float(prediction_linear_regression),
        'prediction_svr': float(prediction_svr),
        'prediction_random_forest_regressor': float(prediction_random_forest_regressor),
        'prediction_gradient_boosting_regressor': float(prediction_gradient_boosting_regressor),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)