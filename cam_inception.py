#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf

from nets import inception_v4
from nets import inception_utils

slim = tf.contrib.slim


number_of_classes = 764


# Define the model that we want to use -- specify to use only two classes at the last layer
def cam_inception(inputs, num_classes=number_of_classes, is_training=True, reuse=None, delta=0.6):

    with tf.variable_scope('InceptionV4',[inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v4.inception_v4_base(inputs, scope=scope)

    inception_c_feature = net
    with tf.variable_scope('cam_classifier/A'):
        net = slim.conv2d(inception_c_feature, 1024, [3, 3],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          padding='SAME',
                          scope='conv1_3x3')
        net = slim.conv2d(net, 1024, [3, 3],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          padding='SAME',
                          scope='conv2_3x3')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          scope='conv3_1x1')
        end_points['features_A'] = net
        # GAP
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
        else:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                 name='global_pool')

        logits = slim.flatten(net, scope='Flatten')
        end_points['Logits'] = logits
        end_points['Predictions_A'] = tf.argmax(logits, 1, name='Predictions_A')

    with tf.variable_scope('cam_classifier/B'):
        batch_size = inception_c_feature.get_shape()[0]
        channels = inception_c_feature.get_shape()[3]
        for n in range(batch_size):
            ca_map = end_points['features_A'][n, :, :, end_points['Predictions_A'][n]]
            ca_map = (ca_map - tf.reduce_min(ca_map)) / (tf.reduce_max(ca_map) - tf.reduce_min(ca_map))
            ca_map = tf.expand_dims(ca_map, 2)
            for i in range(channels):
                if i == 0:
                    erase_tmp = ca_map
                else:
                    erase_tmp = tf.concat([erase_tmp, ca_map], 2)
            erase_tmp = tf.expand_dims(erase_tmp, 0)
            if n == 0:
                erase = erase_tmp
            else:
                erase = tf.concat([erase, erase_tmp], 0)

        erased_feature = tf.where(tf.less(erase, delta), inception_c_feature, tf.zeros_like(inception_c_feature))
        aux_logits = slim.conv2d(erased_feature, 1024, [3, 3],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          padding='SAME',
                          scope='conv1_3x3')
        aux_logits = slim.conv2d(aux_logits, 1024, [3, 3],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          padding='SAME',
                          scope='conv2_3x3')
        aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                          activation_fn=None,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          scope='conv3_1x1')
        end_points['features_B'] = aux_logits
        # GAP
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
            aux_logits = slim.avg_pool2d(aux_logits, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
        else:
            aux_logits = tf.reduce_mean(aux_logits, [1, 2], keep_dims=True,
                                 name='global_pool')

        aux_logits = slim.flatten(aux_logits, scope='Flatten')
        end_points['AuxLogits'] = aux_logits
        end_points['Predictions_B'] = tf.argmax(aux_logits, 1, name='Predictions_B')

    return logits, end_points

def CAMmap(feature_maps, predictions, n_top):
    map_size = feature_maps.get_shape()[1:3]
    heatmap = np.zeros(map_size, n_top)
    tops = np.argsort(-predictions)
    for i in range(n_top):
        feature_map = feature_maps[0, :, :, tops[i]]
        heatmap[:, :, i] = (feature_maps - feature_map.min())/(feature_map.max() - feature_map.min())

    return heatmap

def bounding_box(heatmap, threshold):
    n_boxes = heatmap.shape[2]
    map_size = heatmap.shape[0]
    boxes = np.zeros((n_boxes, 4))
    for i in range(n_boxes):
        ymin, xmin, ymax, xmax = 0, 0, 1, 1
        x1, x2, y1, y2 = False, False, False, False
        for j in range(map_size):
            if x1 == True & y1 == True:
                break
            for k in range(map_size):
                if (heatmap[j, k, i] >= threshold) & (y1 == False):
                    ymin = j / map_size
                    y1 = True
                if (heatmap[k, j, i] >= threshold) & (x1 == False):
                    xmin = j / map_size
                    x1 = True
        for j in reversed(range(map_size)):
            if x2 == True & y2 == True:
                break
            for k in range(map_size):
                if (heatmap[j, k, i] >= threshold) & (y2 == False):
                    ymax = (j + 1) / map_size
                    y2 = True
                if (heatmap[k, j, i] >= threshold) & (x2 == False):
                    xmax = (j + 1) / map_size
                    x2 = True
        bbox = [ymin, xmin, ymax, xmax]
        boxes[i, :] = bbox

    return boxes

cam_inception.default_image_size = 299
cam_inception_arg_scope = inception_utils.inception_arg_scope

