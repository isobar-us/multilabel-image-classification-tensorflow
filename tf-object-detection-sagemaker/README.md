# Training Tensorflow Object Detection Models in AWS Sagemaker
AWS Sagemaker allows us to bring our own algorithms for training. In our case we're planning to train a model to detect objects inside an image.
In order to accomplish this, we'll need to use Tensorflow's object detection API. Since Safemaker doesn't come with Tensforflow
out of box we'll need to deploy our algorithm and its resources inside a Docker image.

## Preparing dataset and Configuring the Training Algorithm

#### Generating the boundary boxes on dataset
Before we start, we have to create boundary boxes for every object in every image of our dataset. 
We used [BBox-Label-Tool](https://github.com/xiaqunfeng/BBox-Label-Tool) to generate our boundary boxes.
This tool generates a txt file for each image to hold the location of objects in that image and looks something like
this:

```text
2
16 969 3527 2246 car
1189 619 2708 1356 person
```

#### Generating `TFRecord` and `label_map.pbtxt`
Our object detection algorithm requires our dataset to be in `TFRecords` format. We'll need one for our training set
`train.records` and another for our validation set `validation.records`. In addition to these files, we also need to
specify our labels in `label_map.pbtxt`. You can use `tfrecord_generator.py` under /resources to generate all three files:

```bash
python resources/tfrecord_generator.py --dataset_base=my_dataset_base --label_path=labels.txt
```
Assuming your dataset has the following layout:
```
/<my_dataset_base>
-> /train
----> /images
-------> image-1.jpg
-------> image-2.jpg
-> /validation
----> /images
-------> image-1.jpg
-------> image-2.jpg
-> /bbox
----> image-1.txt
----> image-2.txt
```

The three file will be generated under `<my_dataset_base>/tf_data`.

#### Create `configuration.config` to represent our object detection configurations
Tensorflow object detection API requires a configuration file to provide the parameters for the training algorithm.
In it, you can specify which image recognition algorithm to use, the number of classes, number of images per batch, etc.
We used MobileNets-v2 for training a model that is optimized to run on a mobile device.

```text
model {
  ssd {
    num_classes: 2
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
      }
    }
    feature_extractor {
      type: "ssd_mobilenet_v2"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 3.99999989895e-05
          }
        }
        initializer {
          random_normal_initializer {
            mean: 0.0
            stddev: 0.00999999977648
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.97000002861
          center: true
          scale: true
          epsilon: 0.0010000000475
        }
      }
      override_base_feature_extractor_hyperparams: true
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 3.99999989895e-05
            }
          }
          initializer {
            truncated_normal_initializer {
              mean: 0.0
              stddev: 0.0299999993294
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.999700009823
            center: true
            scale: true
            epsilon: 0.0010000000475
            train: true
          }
        }
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.800000011921
        kernel_size: 1
        box_code_size: 4
        apply_sigmoid_to_scores: false
      }
    }
    anchor_generator {
      ssd_anchor_generator {
        num_layers: 6
        min_scale: 0.20000000298
        max_scale: 0.949999988079
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.333299994469
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 0.300000011921
        iou_threshold: 0.600000023842
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid {
        }
      }
      hard_example_miner {
        num_hard_examples: 3000
        iou_threshold: 0.990000009537
        loss_type: CLASSIFICATION
        max_negatives_per_positive: 3
        min_negatives_per_image: 0
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
  }
}
train_config {
  batch_size: 34
  data_augmentation_options {
    ssd_random_crop {
    }
  }
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 0.0133330002427
          schedule {
            step: 300
            learning_rate: .01
          }
          schedule {
            step: 600
            learning_rate: .001
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "/opt/ml/input/data/checkpoint/model.ckpt"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader: {
  tf_record_input_reader {
    input_path: "/opt/ml/input/data/train/train.records"
  }
  label_map_path: "/opt/ml/input/data/label/label_map.pbtxt"
}
eval_config {
  num_examples: 24
  # max_evals: 10
  use_moving_averages: false
}
eval_input_reader: {
  tf_record_input_reader {
    input_path: "/opt/ml/input/data/validation/validation.records"
  }
  label_map_path: "/opt/ml/input/data/label/label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
graph_rewriter {
  quantization {
    delay: 1800
    weight_bits: 8
    activation_bits: 8
  }
}
```
#### Using a pre-trained checkpoint
For most of us, we don't want to train from scratch as that would take a long. In that case, we can provide a pre-trained 
checkpoint to our training algorithm to start from. [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
has a list of pre-trained models for us to download. You can add a reference to the checkpoint files inside your
`configuration.config` file:

```text
  fine_tune_checkpoint: "/opt/ml/input/data/checkpoint/model.ckpt"
  from_detection_checkpoint: true
```

#### Upload configuration inputs to AWS S3
In order for us to easily access our configuration and training data, we'll need to upload the following files to S3:
* `train.records`
* `validation.records`
* `configuration.config`
* `label_map.pbtxt`
* and our pre-trained checkpoint (ex. `ssd_mobilenet_v2_quantized_300x300_coco_2018_09_14`).

## Preparing the Sagemaker Docker Container

#### Building the Sagemaker Docker Image
We'll be using Docker to install Tensorflow API and all its requirements for training object detection models. 
Our `Dockerfile` chooses the proper version Tensorflow and builds and installs the object detection API. When the environment
is ready, we run some tests to makes everything installed properly.

#### Deploying the Sagemaker Docker Image to AWS
After building our Docker image, we'll need to deploy it to AWS as ECR. The ERC path can later be referenced inside our
Sagemaker training job. `build_and_push.sh` automates the process of building and deploying the Docker image.

```bash
./build_and_push.sh tf_object_detection_container
```

## AWS Sagemaker Training Job
We're now ready to create a new training job in Sagemaker and point it to our training data and config.

#### Provide container ECR path
When creating the job, we need it to use our Docker image that we uploaded to ECR. Under `Provide container ECR path`
you can provide the path to the Docker image in ECR.

#### Configure Input Data Channels
We'll need to provide the path to our training and configuration files that we uploaded to S3. Configure the following channels:
* `train`: points to `train.records` 
* `validation`: points to `validation.records` 
* `config`: points to `configuration.config`,
* `label`: points to `label_map.pbtxt`
* `checkpoint`: points to `ssd_mobilenet_v2_quantized_300x300_coco_2018_09_14`

You can choose `application/x-image` for content type and `S3Prefix` for S3 data type (ex. `s3://bucket/train.records`).

#### Configure Hyper Parameters
* `num_steps`: number of steps to train the model (ex. 1000)
* `quantize`: whether the model should also generate a TFLite model to run on mobile devices (ex. True)

#### Configure the output S3 folder
Under `Output data configuration` provide the path to the S3 folder for dropping the models after the training is complete.
If training is successful, you should end up with a `.pb` model. If you chose `quantize` you will also have a `.tflite` model.

## Inference
After the training, your model artifacts will be stored in S3 as configured earlier on. The tar.gz output should include 
the model file `frozen_inference_graph.pb`.

#### Loading and preparing the model for inference
Un-tar the output and place it somewhere locally. Now you can use `tf_graph_evaluator.py` script to visualize your model:

```bash
python resources/tf_graph_evaluator.py --frozen_graph_path=frozen_graph.pb --label_path=labels.pbtxt --image_path=test.jpg
```
