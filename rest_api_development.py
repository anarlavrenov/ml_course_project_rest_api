import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import flask
from time import strftime
import os
import dill


def target_multibinary_inversion(x):
    if x == 1:
        return 'POLITICS'
    elif x == 2:
        return 'WELLNESS'
    elif x == 3:
        return 'ENTERTAINMENT'
    elif x == 4:
        return 'TRAVEL'
    elif x == 5:
        return 'STYLE & BEAUTY'
    elif x == 6:
        return 'PARENTING'
    elif x == 7:
        return 'FOOD & DRINK'
    elif x == 8:
        return 'BUSINESS'


model_path = "logreg_pipeline.dill"

app = flask.Flask(__name__)

handler = RotatingFileHandler(filename='logs.txt', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

with open(model_path, 'rb') as f:
    model = dill.load(f)


@app.route("/predict", methods=['POST'])
def give_prediction():
    data = dict(success=False)

    dt = strftime("[%Y-%b-%d %H:%M:%S]")

    if flask.request.method == 'POST':

        headline, short_description = '', ''

        request_json = flask.request.get_json()

        if request_json['headline']:
            headline = request_json['headline']

        if request_json['short_description']:
            short_description = request_json['short_description']

        X_test = pd.DataFrame({'headline': [headline],
                               'short_description': [short_description]})

        prediction = model.predict(X_test)
        prediction = pd.Series(prediction).apply(target_multibinary_inversion)[0]

        data['prediction'] = prediction
        data['success'] = True

        logger.info(f"date: {dt}, success: {data['success']}, prediction: {data['prediction']}")

    return flask.jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8380))
    app.run(host='127.0.0.1', debug=True, port=port)
