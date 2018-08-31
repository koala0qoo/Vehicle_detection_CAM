import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from preprocessing import inception_preprocessing
from nets import cam_inception

from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 764


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--inference_size', type=int, default=1)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


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


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    #PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'exported_graphs/frozen_inference_graph.pb')
    PATH_TO_CKPT = os.path.join(FLAGS.dataset_dir, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'labels.txt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = {}
    label_map = open(PATH_TO_LABELS, 'r', encoding='utf-8')
    for line in label_map:
        cat = {}
        id = line.strip().split(":")[0]
        name = line.strip().split(":")[1]
        cat['id'] = id
        cat['name'] = name
        category_index[int(id)] = cat


    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    #test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('input:0')
            logits = detection_graph.get_tensor_by_name('cam_classifier/A/Flatten/flatten/Reshape:0')
            feature_maps_A = detection_graph.get_tensor_by_name('cam_classifier/A/conv3_1x1/Conv2D:0')
            auxlogits = detection_graph.get_tensor_by_name('cam_classifier/B/Flatten/flatten/Reshape:0')
            feature_maps_B = detection_graph.get_tensor_by_name('cam_classifier/B/conv3_1x1/Conv2D:0')
            for i in range(FLAGS.inference_size):
                image = Image.open(os.path.join(FLAGS.dataset_dir, 'test_{0}.jpg'.format(i)))
                height = image_tensor.get_shape()[1]
                width = image_tensor.get_shape()[2]
                image_resize = image.resize((width,height),Image.BILINEAR)
                image_np = load_image_into_numpy_array(image)
                image_resize_np = load_image_into_numpy_array(image_resize)
                #image_resize_np = (image_resize_np / 255 - 0.5) * 2
                #if image.dtype != tf.float32:      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                #image_resize_np = tf.subtract(image, 0.5)  image_resize_np = tf.multiply(image, 2.0)

                image_np_expanded = np.expand_dims(image_resize_np, axis=0)
                (predictions_1, feature_maps_1, predictions_2, feature_maps_2) = sess.run(
                    [logits, feature_maps_A, auxlogits, feature_maps_B],
                    feed_dict={image_tensor: image_np_expanded})
                predictions_1 = np.squeeze(predictions_1)
                softmax = np.exp(predictions_1)/np.sum(np.exp(predictions_1),axis=0)
                #prediction = np.argmax(predictions)

                n_top = 1
                classes = np.argsort(-predictions_1)[:n_top]
                scores = -np.sort(-softmax)[:n_top]
                print(classes)

                # 生成heatmap
                cam_A = cam_inception.CAMmap(feature_maps_1, predictions_1, n_top)
                cam_B = cam_inception.CAMmap(feature_maps_2, predictions_1, n_top)
                cam = np.maximum(cam_A, cam_B)
                print(cam)
                (im_width, im_height) = image.size
                cam_resize = bilinear(cam, im_height, im_width)

                # 保存heatmap
                for j in range(n_top):
                    heatmap = cam_resize[:, :, j]
                    print(heatmap)
                    heatmap = grey2rainbow(heatmap * 255)
                    heatmap = Image.fromarray(heatmap.astype('uint8')).convert('RGB')
                    heatmap.save(os.path.join(FLAGS.output_dir, 'test_{0}_heatmap_{1}.jpg'.format(i, j)))

                # 生成bounding_boxes
                threshold = 0.9
                boxes = cam_inception.bounding_box(cam_resize, threshold)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes.astype(np.int32),
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=.1,
                    line_thickness=6)
                plt.imsave(os.path.join(FLAGS.output_dir, 'test_{0}_output.jpg'.format(i)), image_np)
