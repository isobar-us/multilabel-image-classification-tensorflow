import argparse

from utils.tf_graph_util import TFGraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frozen_graph_path',
        type=str,
        default='',
        help='Path to the TF frozen graph. It needs to be a .pb file.'
    )
    parser.add_argument(
        '--label_path',
        type=str,
        default='',
        help='Path to labels file. It needs to be in .pbtxt format.'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='Path to the image being evaluated.'
    )
    parser.add_argument(
        '--image_size',
        type=str,
        default='300',
        help='Resize image to this size before inference'
    )
    parser.add_argument(
        '--min_score_thresh',
        type=str,
        default='.3',
        help='Minimum score threshold used for visualization.'
    )
    parser.add_argument(
        '--visualize',
        type=str,
        default='True',
        help='True or False. If True it displays the image with the detected boundary boxes.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    frozen_graph_path = FLAGS.frozen_graph_path
    label_path = FLAGS.label_path
    image_path = FLAGS.image_path
    visualize = str(FLAGS.visualize).lower() == 'true'
    min_score_thresh = float(FLAGS.min_score_thresh)
    image_size = int(FLAGS.image_size)

    print('Frozen graph path set to {}'.format(frozen_graph_path))
    print('Label path set to {}'.format(label_path))
    print('Image path set to {}'.format(image_path))
    print('Visualize flag set to {}'.format(visualize))
    print('Minimum score threshold set to {}'.format(min_score_thresh))
    print('Image size set to {}'.format(image_size))

    tf_graph = TFGraph(label_path, frozen_graph_path)

    if visualize:
        tf_graph.visualize_inference_for_single_image_from_path(image_path=image_path,
                                                                min_score_thresh=min_score_thresh,
                                                                image_size=image_size)
    else:
        output_dict = tf_graph.run_inference_for_single_image_from_path(image_path=image_path,
                                                                        image_size=image_size)
        print(output_dict)


