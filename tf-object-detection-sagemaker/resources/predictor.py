import os
import logging
import json
import base64

import flask

from utils.tf_graph_util import TFGraph

prefix = '/opt/ml/'
input_path = os.path.join(prefix, 'input/data')
model_path = os.path.join(prefix, 'model')
frozen_graph_path = os.path.join(model_path, 'graph/frozen_inference_graph.pb')
label_path = os.path.join(model_path, 'graph/label_map.pbtxt')
param_path = os.path.join(model_path, 'graph/hyperparameters.json')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    graph = None  # Where we keep the model when it's loaded
    training_params = None

    @classmethod
    def get_training_params(cls):
        if cls.training_params is None:
            with open(param_path, 'r') as tc:
                cls.training_params = json.load(tc)

        return cls.training_params

    @classmethod
    def get_graph(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.graph is None:
            cls.graph = TFGraph(label_path, frozen_graph_path)

        return cls.graph

    @classmethod
    def predict(cls, image_data, image_size):
        """For the input, do the predictions and return them."""

        tf_graph = cls.get_graph()
        inference_result = tf_graph.run_inference_for_single_image_from_bytes(image_data, image_size)

        return inference_result


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    print('Ping received...')

    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_graph() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invoke():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """

    logging.info('Invoked with content_type {}'.format(flask.request.content_type))

    if flask.request.content_type == 'application/json':
        logging.info('Running inference on image...')

        body = json.loads(flask.request.data)
        image_data = base64.b64decode(body['image'])

        if 'threshold' in body:
            threshold = body['threshold']
        else:
            threshold = 0.0

        training_params = ScoringService.get_training_params()

        image_size = int(training_params['image_size'])

        # run inference on image
        inference_result = ScoringService.predict(image_data, image_size)

        # convert inference result to json
        prediction_objects = []
        predictions = {}
        detection_classes = inference_result['detection_classes']
        detection_boxes = inference_result['detection_boxes']
        detection_scores = inference_result['detection_scores']
        for index, detection_class in enumerate(detection_classes):
            detection_box = detection_boxes[index].astype(float)
            detection_score = float(detection_scores[index])
            if detection_score < threshold:
                continue

            ymin = detection_box[0]
            xmin = detection_box[1]
            ymax = detection_box[2]
            xmax = detection_box[3]

            prediction_object = [float(detection_class - 1), detection_score, xmin, ymin, xmax, ymax]

            prediction_objects.append(prediction_object)

        predictions['prediction'] = prediction_objects

        return flask.Response(response=json.dumps(predictions), status=200, mimetype='application/json')


    return flask.Response(response="{\"reason\" : \"Request is not application/x-image\"}", status=400, mimetype='application/json')
