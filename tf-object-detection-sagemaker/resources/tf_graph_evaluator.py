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
        '--visualize',
        type=bool,
        default=True,
        help='If true it displays the image with the detected boundary boxes.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    frozen_graph_path = FLAGS.frozen_graph_path
    label_path = FLAGS.label_path
    image_path = FLAGS.image_path
    visualize = FLAGS.visualize

    print('Frozen graph path set to {}'.format(frozen_graph_path))
    print('Label path set to {}'.format(label_path))
    print('Image path set to {}'.format(image_path))
    print('Visualize flag set to {}'.format(visualize))

    tf_graph = TFGraph(label_path, frozen_graph_path)

    if visualize:
        tf_graph.visualize_inference_for_single_image(image_path)
    else:
        output_dict = tf_graph.run_inference_for_single_image(image_path)
        print(output_dict)


