# Multi-label Image Classification with Tensorflow
This project modifies the existing `retrain.py` file from Tensorflow Hub and enables it to do multi-label classification on images using a given pre-trained feature vector module.

`label_image.py` is a modified version of Tensorflow's example labeling script to accept different image resizing methods.

### How to use
Follow these steps to train and label your dataset.

#### Preparing your dataset of images
The trainer expects all your images to exist inside a folder called `images` within your main directory. In addition to this folder, you'll need to provide a LST file for your training set called `dataset_train.lst` and a LST file for your validation set called `dataset_validation.lst`. These files need to exist in your main directory at the same level as your images folder.

```
/path-to-dataset
--> dataset_train.lst
--> dataset_validation.lst
--> /images
----> image-1.jpg
----> image-2.jpg
```

##### How to build a Lst file
Your list file should contain a unique image index, followed by a tab-separated list of binary flags for each label (`1` to indicate label exists in image, and `0` to indicate label doesn't exist), followed by a relative path to the image.

Here's an example for a 5-label classification:
```
2645	0	0	1	0	1	images/13561986193_cf645b2b9a.jpg	
1090	1	0	0	1	1	images/21652746_cc379e0eea_m.jpg
15	0	1	0	1	0	images/9558628596_722c29ec60_m.jpg
3008	1	0	1	0	1	images/17720403638_94cfcd8d5c_n.jpg
...
```

Make sure your items in the LST file are shuffled and note that each column is separated by a tab.

#### Training
Here we'll train a pre-trained model with our own dataset.

##### Selecting a feature vector module
Select a feature vector module from [TensorFlow Hub](https://tfhub.dev/s?module-type=image-feature-vector&publisher=google). You could select MobileNet, Inception, ResNet, etc.

##### Specify number of labels for classification
You'll need to specify how many labels are being used during the classification. Pass in the total count via `--num_classes` the training script

##### Specify percentage for testing (optional)
After the training is finished, the script uses a portion of your validation dataset to perform a final evaluation of your model. You can specify the testing percentage `--testing_percentage` or the script assumes 10% by default.

##### Example
In this example we're training on MobileNet for a 1000 steps for a 5-label classification:

```
python retrain.py --image_dir ./dataset-flower-multi/ --how_many_training_steps=1000 --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2 --num_classes=5
```

#### Inference
Time to classify your images

##### Specify an image resizing method (optional)
By default script uses `tf.image.ResizeMethod.AREA` for resizing your images before passing them to the model. Using `--resize_method` you can select from `AREA, NEAREST_NEIGHBOR, BILINEAR, BICUBIC`

#### Example
```
python label_image.py --input_height=224 --input_width=224 --graph=/tmp/output_graph.pb --labels=labels.txt --input_layer=Placeholder --output_layer=final_result --image=image.jpg
```

		

