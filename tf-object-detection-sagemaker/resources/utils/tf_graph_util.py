import numpy as np
import io
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_image_into_numpy_array_from_path(image_path, image_size):
    file_name = image_path

    image = Image.open(file_name)
    image = image.resize((image_size, image_size), Image.ANTIALIAS)
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    return image_np


def load_image_into_numpy_array_from_bytes(image_bytes, image_size):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((image_size, image_size), Image.ANTIALIAS)
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    return image_np


class TFGraph:
    def __init__(self, label_path, frozen_graph_path):
        self.label_path = label_path
        self.frozen_graph_path = frozen_graph_path

        self._load_graph()

        print('Loaded model and labels.')

    def _load_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default() as default_graph:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        self.category_index = label_map_util.create_category_index_from_labelmap(self.label_path, use_display_name=True)
        self.session = tf.Session(config=config, graph=default_graph)
        self.global_graph = default_graph

    def _run_inference_for_single_image(self, image_numpy):
        # Get handles to input and output tensors
        ops = self.global_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.global_graph.get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_numpy.shape[0], image_numpy.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

        image_tensor = self.global_graph.get_tensor_by_name('image_tensor:0')

        # Run inference
        expanded_image_numpy = np.expand_dims(image_numpy, 0)
        print('Running inference for image...')
        output_dict = self.session.run(tensor_dict, feed_dict={image_tensor: expanded_image_numpy})
        print('Finished running inference')
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def run_inference_for_single_image_from_bytes(self, image_bytes, image_size=300):
        image_np = load_image_into_numpy_array_from_bytes(image_bytes, image_size)

        return self._run_inference_for_single_image(image_np)

    def run_inference_for_single_image_from_path(self, image_path, image_size=300):
        image_np = load_image_into_numpy_array_from_path(image_path, image_size)

        return self._run_inference_for_single_image(image_np)

    def visualize_inference_for_single_image_from_path(self, image_path,
                                                       min_score_thresh=.3,
                                                       line_thickness=4,
                                                       output_image_size=(12, 8),
                                                       image_size=300):
        image_np = load_image_into_numpy_array_from_path(image_path, image_size)

        # Actual detection.
        output_dict = self._run_inference_for_single_image(image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            min_score_thresh=min_score_thresh,
            line_thickness=line_thickness)
        plt.figure(figsize=output_image_size)
        plt.imshow(image_np)
        plt.show()
