from flask import Flask, jsonify, json, request
from flask_cors import CORS
import os
from predict import get_knowledge_points,get_l1_symbols,get_l2_symbols,get_l3_symbols,get_l4_symbols

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, resources={r"/*": {'origins': '*'}})
CORS(app, resources={r"/*": {'origins': 'http://localhost:8080', 'allow_headers':
    'Access-Control-Allow-Origin'}})
PROBLEMS_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", "problems.json")


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/problems', methods=['GET'])
def get_problems():  # put application's code here
    with open(PROBLEMS_ROOT, 'r', encoding="utf8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/predict', methods=['POST'])
def predict():  # put application's code here
    question = request.get_data(as_text=True)
    print('--------------------')
    print(question)
    kps = get_knowledge_points(question)
    l1_data = get_l1_symbols(question)
    l2_data = get_l2_symbols(question)
    l3_data = get_l3_symbols()
    l4_data = get_l4_symbols()
    # l3_data = get_l3_symbols(question)
    # l4_data = get_l4_symbols(question)
    # data = predict.get_prediction()
    print(kps)
    return jsonify({
        'kps': kps,
        'l1_data': l1_data,
        'l2_data': l2_data,
        'l3_data': l3_data,
        'l4_data': l4_data,
    })


if __name__ == '__main__':
    app.run()
