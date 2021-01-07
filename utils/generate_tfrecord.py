"""
Usage:

# Create train an test data:
python generate_tfrecord.py --train_size=0.7 --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv --img_path=<PATH_TO_IMAGES> --output_path=<PATH_TO_OUTPUT_FOLDER>

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import absl

from PIL import Image
from numpy.random.mtrand import RandomState
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = absl.flags
flags.DEFINE_float('train_size', 0.7, 'Percentage of the data set used for the training data')
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
# for multiple labels add more else if statements
def class_text_to_int(row_label):
    if row_label == 'GER':
        return 1
    # comment upper if statement and uncomment these statements for multiple labelling
    # if row_label == FLAGS.label0:
    #   return 1
    # elif row_label == FLAGS.label1:
    #   return 0
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(name, examples, img_path):
    writer = tf.io.TFRecordWriter(os.path.join(FLAGS.output_path, name))

    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, img_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('{} images stored in TFRecords: {}'.format(len(examples), os.path.join(FLAGS.output_path, name)))


def main(_):
    img_path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    print('{} images found'.format(len(examples)))

    #  train / test split
    rng = RandomState()
    train = examples.sample(frac=FLAGS.train_size, random_state=rng)
    test = examples.loc[~examples.index.isin(train.index)]

    create_tf_record('train.tfrecord', train, img_path)
    create_tf_record('eval.tfrecord', test, img_path)


if __name__ == '__main__':
    tf.compat.v1.app.run()
