# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

is_training_placeholder = tf.placeholder(tf.bool)

def grey2rainbow(grey):
    h, w = grey.shape
    rainbow = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if grey[i, j] <= 51:
                rainbow[i, j, 0] = 255
                rainbow[i, j, 1] = grey[i, j] * 5
                rainbow[i, j, 2] = 0
            elif grey[i, j] <= 102:
                rainbow[i, j, 0] = 255 - (grey[i, j] - 51) * 5
                rainbow[i, j, 1] = 255
                rainbow[i, j, 2] = 0
            elif grey[i, j] <= 153:
                rainbow[i, j, 0] = 0
                rainbow[i, j, 1] = 255
                rainbow[i, j, 2] = (grey[i, j] - 102) * 5
            elif grey[i, j] <= 204:
                rainbow[i, j, 0] = 0
                rainbow[i, j, 1] = 255 - int((grey[i, j] - 153) * 128 / 51 + 0.5)
                rainbow[i, j, 2] = 255
            elif grey[i, j] <= 255:
                rainbow[i, j, 0] = 0
                rainbow[i, j, 1] = 127 - int((grey[i, j] - 204) * 127 / 51 + 0.5)
                rainbow[i, j, 2] = 255

    return rainbow


def bilinear(img, h, w):
    height, width, channels = img.shape
    if h == height and w == width:
        return img
    new_img = np.zeros((h, w, channels), np.uint8)
    scale_x = float(width) / w
    scale_y = float(height) / h
    for n in range(channels):
        for dst_y in range(h):
            for dst_x in range(w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, width - 1)
                src_y_1 = min(src_y_0 + 1, height - 1)

                value0 = (src_x_1 - src_x) * img[src_y_0, src_x_0, n] + (src_x - src_x_0) * img[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * img[src_y_1, src_x_0, n] + (src_x - src_x_0) * img[src_y_1, src_x_1, n]
                new_img[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)

    return new_img


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)
    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    checkpoint_path = FLAGS.checkpoint_path
    feed_dict_to_use = {is_training_placeholder: True}

  with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()
    sess.run(init_op)
    sess.run(init_local_op)
    variables_to_restore = slim.get_model_variables()
    saver = tf.train.Saver(max_to_keep=5)
    saver.restore(sess, checkpoint_path)
    logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))
    #init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
    #init_fn(sess)
    for i in range(40):
        predictions_1, feature_map_a, feature_map_b = sess.run([logits, end_points['features_A'], end_points['features_B']], feet_dict=feed_dict_to_use)
        predictions_1 = np.squeeze(predictions_1)

        n_top = 1
        classes = np.argsort(-predictions_1)[:n_top]
        softmax = np.exp(predictions_1) / np.sum(np.exp(predictions_1), axis=0)
        scores = -np.sort(-softmax)[:n_top]

        # 生成heatmap
        cam_A = cam_inception.CAMmap(feature_map_a, predictions_1, n_top)
        cam_B = cam_inception.CAMmap(feature_map_b, predictions_1, n_top)
        cam = np.maximum(cam_A, cam_B)
        (im_width, im_height) = image.size
        cam_resize = bilinear(cam, im_height, im_width)

        # 保存heatmap
        for j in range(n_top):
            heatmap = cam_resize[:, :, j]
            heatmap = grey2rainbow(heatmap * 255)
            heatmap = Image.fromarray(heatmap.astype('uint8')).convert('RGB')
            heatmap.save(os.path.join(FLAGS.output_dir, 'test_images/test_{0}_heatmap_{1}.jpg'.format(i, j)))

        # 生成bounding_boxes
        threshold = 0.5
        boxes = cam_inception.bounding_box(cam_resize, threshold)
        
        PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'labels.txt')

        category_index = {}
        label_map = open(PATH_TO_LABELS, 'r', encoding='utf-8')
        for line in label_map:
            cat = {}
            id = line.strip().split(":")[0]
            name = line.strip().split(":")[1]
            cat['id'] = id
            cat['name'] = name
            category_index[int(id)] = cat       
        

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes.astype(np.int32),
            scores,
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=.3,
            line_thickness=6)
        plt.imsave(os.path.join(FLAGS.output_dir, 'test_images/test_{0}_output.jpg'.format(i)), image_np)


if __name__ == '__main__':
  tf.app.run()
