# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    print(image_reader)
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    sess = tf.Session()
    result = sess.run(normalized)
    # print(result)
    # print()
    # img_array2 = cv2.imread(os.path.join(path2,img), cv2.IMREAD_GRAYSCALE)
    # img_array = cv2.imread(file_name)
    # img_array = tf.cast(img_array, tf.float32)
    # resized_image_img_array = cv2.resize(img_array, (input_height, input_width))
    # resized_image_img_array = tf.expand_dims(resized_image_img_array, axis=0) # expand the dimension
    #
    # plt.imshow(resized_image_img_array)
    # plt.show()
    return result


def old_read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    sess = tf.Session()
    result = sess.run(normalized)
    # print(result)
    # print()
    plt.imread(file_name)
    plt.show()
    return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
    file_name = "/Users/huyvu/OneDrive/robotics/E_frame012.jpg"
    file_name2 = "/Users/huyvu/OneDrive/robotics/F.jpg"
    model_file = "./output_graph.pb"
    label_file = "./output_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=file_name,
        help="image to be processed")
    parser.add_argument(
        "--graph",
        default=model_file,
        help="graph/model to be executed")
    parser.add_argument(
        "--labels",
        default=label_file,
        help="name of file containing labels")
    parser.add_argument(
        "--input_height",
        type=int,
        default= input_height,
        help="input height")
    parser.add_argument(
        "--input_width",
        type=int,
        default = input_width,
        help="input width")
    parser.add_argument(
        "--input_mean",
        type=int,
        default=input_mean,
        help="input mean")
    parser.add_argument(
        "--input_std",
        type=int,
        default = input_std,
        help="input std")
    parser.add_argument(
        "--input_layer",
        default = input_layer,
        help="name of input layer")
    parser.add_argument(
        "--output_layer",
        default=output_layer,
        help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
    print(t)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    print(label_file)
    labels = load_labels(label_file)
    for i in top_k[:2]:
        print(labels[i], results[i])
    print()

    t2 = read_tensor_from_image_file(
        file_name2,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t2
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k[:2]:
        print(labels[i], results[i])

# run test
# python3.6 label_image.py \
# --output_layer=final_result \
