# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
FROM tensorflow/tensorflow:1.11.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends git curl

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /resources /opt/ml/code
WORKDIR /opt/ml/code

# Install Tensorflow's Object Detection API
RUN pip install --upgrade pip
RUN pip install matplotlib

RUN cd tensorflow-models/research/slim \
    pip install -e .

RUN cd tensorflow-models/research \
    protoc object_detection/protos/*.proto --python_out=. \
    python setup.py build \
    python setup.py install

ENV PYTHONPATH=${PYTHONPATH}:tensorflow-models/research:tensorflow-models/research/slim:tensorflow-models/research/object_detection

RUN pip install -U scikit-image
RUN pip install -U cython
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

# Test Tensorflow's Object Detection API
RUN python tensorflow-models/research/object_detection/builders/model_builder_test.py
