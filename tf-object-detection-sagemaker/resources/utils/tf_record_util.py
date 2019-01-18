import os
import random

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util


def shuffle_dataset(files, shuffle_count):
    for i in range(0, shuffle_count):
        random.shuffle(files)


class TfRecordGenerator:

    def __init__(self,
                 bbox_dir,
                 dataset_dir,
                 dataset_train_dir,
                 dataset_validation_dir,
                 labels):
        self.bbox_dir = bbox_dir
        self.dataset_dir = dataset_dir
        self.dataset_train_dir = dataset_train_dir
        self.dataset_validation_dir = dataset_validation_dir
        self.labels = labels

    def _create_tf_example(self, image_dir, image_filename):

        image_name = os.path.splitext(image_filename)[0]
        image_format = os.path.splitext(image_filename)[1]

        image_path = image_dir + "/" + image_filename

        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            image_data = bytearray(image_data)

        img = Image.open(image_path)
        width, height = img.size

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        with open(self.bbox_dir + "/" + image_name + ".txt", 'r') as annotation_file:
            annotation_lines = annotation_file.readlines()
            annotation_lines.pop(0)
            for annotation_line in annotation_lines:
                coords = annotation_line.strip().split(" ")
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
                annotation_label = coords[4]
                annotation_class_id = self.labels.index(annotation_label)

                xmins.append(x1 / width)
                xmaxs.append(x2 / width)
                ymins.append(y1 / height)
                ymaxs.append(y2 / height)
                classes_text.append(bytes(annotation_label, 'utf-8'))
                classes.append(annotation_class_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(bytes(image_filename, 'utf-8')),
            'image/source_id': dataset_util.bytes_feature(bytes(image_filename, 'utf-8')),
            'image/encoded': dataset_util.bytes_feature(bytes(image_data)),
            'image/format': dataset_util.bytes_feature(bytes(image_format, 'utf-8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tf_example

    def generate_tf_records(self):

        train_writer = tf.python_io.TFRecordWriter(self.dataset_dir + '/tf_data/train.records')
        validation_writer = tf.python_io.TFRecordWriter(self.dataset_dir + '/tf_data/validation.records')

        files = os.listdir(self.dataset_train_dir + '/images')
        shuffle_dataset(files, 3)
        for file in files:
            if file.endswith('.jpg'):
                tf_example = self._create_tf_example(self.dataset_train_dir + '/images', file)
                train_writer.write(tf_example.SerializeToString())

        train_writer.close()

        files = os.listdir(self.dataset_validation_dir + '/images')
        shuffle_dataset(files, 3)
        for file in files:
            if file.endswith('.jpg'):
                tf_example = self._create_tf_example(self.dataset_validation_dir + '/images', file)
                validation_writer.write(tf_example.SerializeToString())

        validation_writer.close()

    def generate_label_map(self):
        with open(self.dataset_dir + '/tf_data/label_map.pbtxt', 'w') as label_map_file:
            for index, label in enumerate(self.labels):
                label_map_file.write('item {\n')
                label_map_file.write(' id: ' + str(index + 1) + '\n')
                label_map_file.write(" name: '" + label + "'\n")
                label_map_file.write('}\n\n')