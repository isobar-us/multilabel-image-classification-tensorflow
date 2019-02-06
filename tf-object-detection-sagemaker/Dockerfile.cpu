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

FROM tensorflow/tensorflow:1.12.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        zip \
        unzip \
        git \
        ca-certificates \
        curl \
        nginx

#Install protobuff/protoc
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /resources /opt/ml/code
WORKDIR /opt/ml/code

RUN pip install --upgrade pip
RUN pip install matplotlib

#RUN cd tensorflow-models/research/slim \
RUN pip install -e tensorflow-models/research/slim

#Build and install Tensorflow's Object Detection API
WORKDIR tensorflow-models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN python setup.py build
RUN python setup.py install

WORKDIR /opt/ml/code

#Update the python path to include the object detection API
ENV PYTHONPATH=${PYTHONPATH}:tensorflow-models/research:tensorflow-models/research/slim:tensorflow-models/research/object_detection

RUN echo $PYTHONPATH

RUN pip install -U scikit-image
RUN pip install -U scikit-learn
RUN pip install -U flask
RUN pip install -U gevent
RUN pip install -U gunicorn
RUN pip install -U cython
RUN pip install -U scipy
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
