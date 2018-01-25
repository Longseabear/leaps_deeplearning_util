
from __future__ import division
import argparse

import fnmatch
import os
import tensorflow as tf
import cv2
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def dataset_files(rootdir, pattern='*.JPEG'):
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def is_corrupt_image(img):
    """
        check corrupt image
    :param img: rgb image [ uint8 ]
    :return:
    """
    if np.allclose(img[-1, :, :], np.array([128, 128, 128])):
        print('error checking [{}].'.format(i - 1))
        return True
    return False

def write_tfrecord(file_list, tfrecord_name, example_per_file=10000, iter_per_log = 1000, corrupt_check=True):
    """
    :param file_list: all_file_name_list.
    :param tfrecord_name: output_dir + file_name ( not add .tfrecords )
    :param example_per_file: number of example per tfrecords file
    :param iter_per_log: log
    :param corrupt_check: corrupt image check
    :return: none. (save tfrecords file in output path)
    """
    i = 0
    index = 0
    total_file_num = len(file_list)

    writer = tf.python_io.TFRecordWriter(tfrecord_name + '_{}.tfrecords'.format(index))
    print('loading image..')
    for img_path in file_list:

        img = cv2.imread(img_path)

        if corrupt_check:
            if is_corrupt_image(img):
                total_file_num -= 1
                continue

        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
        }))
        writer.write(example.SerializeToString())

        i += 1
        if i % iter_per_log == 0:
            print('processing...', i, '/', total_file_num)

        if i % example_per_file == 0:
            print('saving [ ', index, ' ]')
            index += 1
            if i < total_file_num:
                writer.close()
                writer = tf.python_io.TFRecordWriter(tfrecord_name + '_{}.tfrecords'.format(index))

    print('saving ok..')
    writer.close()

parser = argparse.ArgumentParser(description='Layered Sampling')
parser.add_argument('--dataset_dir', type=str,
                    help='Path to dataset directory.')
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.')
parser.add_argument('--file_name', type=str,
                    help='Path to output directory.')
parser.add_argument('--example_per_file', type=int, default=10000,
                    help='Number of Example Per Tfrecords files')
parser.add_argument('--iter_per_log', type=int, default=1000,
                    help='iter per log')
parser.add_argument('--object_dataset_num', type=int, default=100000,
                    help='total number of object datas')
parser.add_argument('--seed', type=int, default=13,
                    help='random_seed')
parser.add_argument('--corrupt_check', type=bool, default=False,
                    help='corrupt_check function')
FLAGS = parser.parse_args()

dataset_dir = FLAGS.dataset_dir
output_dir = FLAGS.output_dir
file_name = FLAGS.file_name
example_per_file = FLAGS.example_per_file
iter_per_log = FLAGS.iter_per_log

assert(os.path.isdir(dataset_dir))
layered_label = []
for i in os.listdir(dataset_dir):
    if i.find('.') < 0:
        layered_label.append(dataset_dir+'/'+i)

label_num = len(layered_label)
object_dataset_num = FLAGS.object_dataset_num
dataset_per_label = int(object_dataset_num/label_num)
remain = object_dataset_num % label_num

total_file_list = []
for i in range(label_num):
    data = dataset_files(layered_label[i])
    if i < remain:
        total_file_list += data[:dataset_per_label+1]
    else:
        total_file_list += data[:dataset_per_label]
print(len(total_file_list))

np.random.seed(FLAGS.seed)
np.random.shuffle(total_file_list)
write_tfrecord(total_file_list, output_dir+file_name,example_per_file=example_per_file,
               iter_per_log=iter_per_log,corrupt_check=FLAGS.corrupt_check)