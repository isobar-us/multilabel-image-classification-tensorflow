import os

import flask

from utils.tf_graph_util import TFGraph

prefix = '/opt/ml/'
input_path = os.path.join(prefix, 'input/data')
model_path = os.path.join(prefix, 'model')
frozen_graph_path = os.path.join(model_path, 'graph/frozen_inference_graph.pb')
label_path = os.path.join(model_path, 'graph/label_map.pbtxt')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    graph = None  # Where we keep the model when it's loaded

    @classmethod
    def get_graph(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.graph is None:
            cls.graph = TFGraph(label_path, frozen_graph_path)
        return cls.graph

    @classmethod
    def predict(cls, request):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    print('Invoked with content_type {}'.format(flask.request.content_type))

    if flask.request.content_type == 'application/x-image':
        image_data = flask.request.data

    return flask.Response(response="{\"status\" : \"success\"}", status=200, mimetype='application/json')
