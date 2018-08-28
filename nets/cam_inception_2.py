#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf

import inception_v4
from nets import inception_utils

slim = tf.contrib.slim


upsample_factor = 8
number_of_classes = 764
'''
def normalize(h):
    max =
    min =
    norm =  (h - min) / (max - min)
    return norm
'''

# Define the model that we want to use -- specify to use only two classes at the last layer
def cam_classifier(inputs, num_classes=number_of_classes, is_training=True, delta=0.6):

    with tf.variable_scope('InceptionV4',[inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v4_base(inputs, scope=scope)

    inception_c_feature = net
    with tf.variable_scope('cam_classifier/A'):
        net = slim.conv2d(inception_c_feature, 1024, [3, 3],
                          weights_initializer=tf.zeros_initializer,
                          padding='SAME',
                          scope='conv1_3x3')
        net = slim.conv2d(net, 1024, [3, 3],
                          weights_initializer=tf.zeros_initializer,
                          padding='SAME',
                          scope='conv2_3x3')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          weights_initializer=tf.zeros_initializer,
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

        #map = end_points['features_A'][:, :, :, end_points['Predictions']]？？？
        map = normalize(map)
        end_points['map_A'] = map

    with tf.variable_scope('cam_classifier/B'):
        map_shape = map.get_shape()
        feature_shape = inception_c_feature.get_shape()[0:3]
        threshold = tf.constant(delta, shape=map_shape, dtype=float32, name='threshold')
        zeros = tf.constant(0, shape=feature_shape, dtype=float32, name='zeros')
        erase = tf.greater(end_points['map_A'], threshold, name='compare')
        erased_feature = tf.select(erase, zeros, inception_c_feature, name='erase')
        aux_logits = slim.conv2d(erased_feature, 1024, [3, 3],
                          weights_initializer=tf.zeros_initializer,
                          padding='SAME',
                          scope='conv1_3x3')
        aux_logits = slim.conv2d(aux_logits, 1024, [3, 3],
                          weights_initializer=tf.zeros_initializer,
                          padding='SAME',
                          scope='conv2_3x3')
        aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                          activation_fn=None,
                          weights_initializer=tf.zeros_initializer,
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
        # map = end_points['features_A'][:, :, :, end_points['Predictions']]？？？
        map = normalize(map)
        end_points['map_B'] = map


    return logits, end_points


cam_inception.default_image_size = 299
cam_inception_arg_scope = inception_utils.inception_arg_scope


'''
def CAMmap(last_conv_net, w_variables):



    n_feat, w, h, n = activation.shape 
    act_vec = np.reshape(activation, [n_feat, w * h])
    n_top = weights_LR.shape[0]
    out = np.zeros([w, h, n_top])
    for t in range(n_top):
        weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
        heatmap_vec = np.dot(weights_vec, act_vec)
        heatmap = np.reshape(np.squeeze(heatmap_vec), [w, h])
        out[:, :, t] = heatmap
    return out
'''

def bounding_box(image, heatmap, threshold):



downsampled_logits_shape = tf.shape(logits)

img_shape = tf.shape(image_tensor)

# Calculate the ouput size of the upsampled tensor
# The shape should be batch_size X width X height X num_classes
upsampled_logits_shape = tf.stack([
                                  downsampled_logits_shape[0],
                                  img_shape[1],
                                  img_shape[2],
                                  downsampled_logits_shape[3]
                                  ])


pool4_feature = end_points['vgg_16/pool4']
pool3_feature = end_points['vgg_16/pool3']

with tf.variable_scope('vgg_16/fc8'):
    aux_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool4')
    aux_logits_8s = slim.conv2d(pool3_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool3')


# Perform the upsampling of logits(32s to 16s)
upsample_filter_np_x2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                  number_of_classes)

upsample_filter_tensor_x2_1 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2_1')

upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2_1,
                                          output_shape=tf.shape(aux_logits_16s),
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')

upsampled_logits = upsampled_logits + aux_logits_16s


# Perform the upsampling of upsampled_logits(16s to 8s)
upsample_filter_tensor_x2_2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2_2')

upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x2_2,
                                          output_shape=tf.shape(aux_logits_8s),
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')

upsampled_logits = upsampled_logits + aux_logits_8s


# Perform the final upsampling
upsample_filter_np_x8 = bilinear_upsample_weights(upsample_factor,
                                                   number_of_classes)

upsample_filter_tensor_x8 = tf.Variable(upsample_filter_np_x8, name='vgg_16/fc8/t_conv_x8')
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1, upsample_factor, upsample_factor, 1],
                                          padding='SAME')


lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_logits,
                                                          labels=lbl_onehot)

cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))


# Tensor to get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
pred = tf.argmax(upsampled_logits, axis=3)

probabilities = tf.nn.softmax(upsampled_logits)

# Here we define an optimizer and put all the variables
# that will be created under a namespace of 'adam_vars'.
# This is done so that we can easily access them later.
# Those variables are used by adam optimizer and are not
# related to variables of the vgg model.

# We also retrieve gradient Tensors for each of our variables
# This way we can later visualize them in tensorboard.
# optimizer.compute_gradients and optimizer.apply_gradients
# is equivalent to running:
# train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_loss)
with tf.variable_scope("adam_vars"):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    gradients = optimizer.compute_gradients(loss=cross_entropy_loss)

    for grad_var_pair in gradients:

        current_variable = grad_var_pair[1]
        current_gradient = grad_var_pair[0]

        # Relace some characters from the original variable name
        # tensorboard doesn't accept ':' symbol
        gradient_name_to_save = current_variable.name.replace(":", "_")

        # Let's get histogram of gradients for each layer and
        # visualize them later in tensorboard
        tf.summary.histogram(gradient_name_to_save, current_gradient)

    train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.
vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])

# Here we get variables that belong to the last layer of network.
# As we saw, the number of classes that VGG was originally trained on
# is different from ours -- in our case it is only 2 classes.
vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

# Add summary op for the loss -- to be able to see it in
# tensorboard.
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

# Put all summary ops into one op. Produces string when
# you run it.
merged_summary_op = tf.summary.merge_all()

# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

checkpoint_path = tf.train.latest_checkpoint(log_folder)
continue_train = False
if checkpoint_path:
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % log_folder)
    variables_to_restore = slim.get_model_variables()

    continue_train = True

else:

    # Create an OP that performs the initialization of
    # values of variables to the values from VGG.
    read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
        vgg_checkpoint_path,
        vgg_except_fc8_weights)

    # Initializer for new fc8 weights -- for two classes.
    vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

    # Initializer for adam variables
    optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

init_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)


with sess:
    # Run the initializers.
    sess.run(init_op)
    sess.run(init_local_op)
    if continue_train:
        saver.restore(sess, checkpoint_path)

        logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))
    else:
        sess.run(vgg_fc8_weights_initializer)
        sess.run(optimization_variables_initializer)

        read_vgg_weights_except_fc8_func(sess)
        logging.debug('value initialized...')

    # start data reader
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start = time.time()
    for i in range(FLAGS.max_steps):
        feed_dict_to_use[is_training_placeholder] = True

        gs, _ = sess.run([global_step, train_step], feed_dict=feed_dict_to_use)
        if gs % 10 == 0:
            gs, loss, summary_string = sess.run([global_step, cross_entropy_loss, merged_summary_op], feed_dict=feed_dict_to_use)
            logging.debug("step {0} Current Loss: {1} ".format(gs, loss))
            end = time.time()
            logging.debug("[{0:.2f}] imgs/s".format(10 * batch_size / (end - start)))
            start = end

            summary_string_writer.add_summary(summary_string, i)

            if gs % 100 == 0:
                save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
                logging.debug("Model saved in file: %s" % save_path)

            if gs % 200 == 0:
                eval_folder = os.path.join(FLAGS.output_dir, 'eval')
                if not os.path.exists(eval_folder):
                    os.makedirs(eval_folder)

                logging.debug("validation generated at step [{0}]".format(gs))
                feed_dict_to_use[is_training_placeholder] = False
                val_pred, val_orig_image, val_annot, val_poss = sess.run([pred, orig_img_tensor, annotation_tensor, probabilities],
                                                                         feed_dict=feed_dict_to_use)

                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_img.jpg'.format(gs)), cv2.cvtColor(np.squeeze(val_orig_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_annotation.jpg'.format(gs)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_annot)), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_prediction.jpg'.format(gs)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_pred)), cv2.COLOR_RGB2BGR))

                crf_ed = perform_crf(val_orig_image, val_poss)
                cv2.imwrite(os.path.join(FLAGS.output_dir, 'eval', 'val_{0}_prediction_crfed.jpg'.format(gs)), cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR))

                overlay = cv2.addWeighted(cv2.cvtColor(np.squeeze(val_orig_image), cv2.COLOR_RGB2BGR), 1, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR), 0.8, 0)
                cv2.imwrite(os.path.join(FLAGS.output_dir, 'eval', 'val_{0}_overlay.jpg'.format(gs)), overlay)

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, os.path.join(log_folder, "model.ckpt"), global_step=gs)
    logging.debug("Model saved in file: %s" % save_path)

summary_string_writer.close()
