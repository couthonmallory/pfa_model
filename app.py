import json
from flask import Flask, jsonify, request
import pfa_model as model
import pandas as pd

app = Flask(__name__)

MODEL_TRAINED = model.pfa_model()
MODEL_TRAINED.Train()

@app.route('/', methods=['GET'])
def helloworld():
    data={"data":"Hello World"}
    return data

@app.route('/teste', methods=['POST'])
def teste_function():
    teste_case = json.loads(request.data)
    # Préparation des données
    data = [[teste_case["temperature"], teste_case["humidity"]]]
    # Création du DataFrame
    df = pd.DataFrame(data, columns=['temperature', 'humidity'])
    # models = model.pfa_model()
    # models.Train()
    data = MODEL_TRAINED.test_function(df)
    return jsonify({
        "data" : data.tolist()[0]
    })

@app.route('/home', methods=['GET'])
def getdata():
    return json.dumps({'name': 'alice',
    'email': 'alice@outlook.com'})
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=int("3000") )