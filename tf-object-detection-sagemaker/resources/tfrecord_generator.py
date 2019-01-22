import argparse

from utils.tf_record_util import TfRecordGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_base',
        type=str,
        default='',
        help='Path to dataset base. Should include the /bbox, /train, and /validation folders'
    )
    parser.add_argument(
        '--label_path',
        type=str,
        default='',
        help='Path to a list of labels'
    )
    FLAGS, unparsed = parser.parse_known_args()

    DATASET_BASE = FLAGS.dataset_base

    bbox_dir = DATASET_BASE + "/bbox"
    dataset_train_dir = DATASET_BASE + "/train"
    dataset_validation_dir = DATASET_BASE + "/validation"

    with open(FLAGS.label_path, 'r') as labels_file:
        label_list = [line.strip() for line in labels_file.readlines()]

    print('Loaded labels {}'.format(label_list))

    tf_record_generator = TfRecordGenerator(bbox_dir=bbox_dir,
                                            dataset_dir=DATASET_BASE,
                                            dataset_train_dir=dataset_train_dir,
                                            dataset_validation_dir=dataset_validation_dir,
                                            labels=label_list)

    print('Generating TFRecord files...')
    tf_record_generator.generate_tf_records()

    print('Generating label_map.pbtxt...')
    tf_record_generator.generate_label_map()

    print('Finished generating all files.')

