#!/usr/bin/env python

import json
import os
import sys
import traceback
from shutil import copyfile

import tensorflow as tf
from utils.tf_graph_util import TFGraph
import tensorflow.contrib.tensorrt as trt

from tensorrt.object_detection import optimize_model
from utils import commandline_util

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info(os.listdir('./resnet'))

    tf.logging.info('Converting graph to tfrt')
    frozen_graph = optimize_model(
        config_path=str('/opt/ml/code/resnet/pipeline.config'),
        checkpoint_path=str('/opt/ml/code/resnet/model.ckpt'),
        use_trt=True,
        minimum_segment_size=15,
        precision_mode='FP16',
        output_path=str('/opt/ml/code/resnet/optimized_frozen_inference_graph.pb')
    )

    tf.logging.info('Now loading model')

    tf_graph = TFGraph('/opt/ml/code/resnet/label_map.pbtxt', '/opt/ml/code/resnet/optimized_frozen_inference_graph.pb')

    tf.logging.info('Now running inference')
    inference_result = tf_graph.run_inference_for_single_image_from_path('/opt/ml/code/resnet/green-kingfisher.jpg',
                                                                         640)

    tf.logging.info(inference_result)
