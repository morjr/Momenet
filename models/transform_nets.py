import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def input_transform_net(point_cloud, is_training, bn_decay=None, K=9):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 128, [1,9],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    #net = tf_util.conv2d(net, 128, [1,1],
    #                     padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training,
    #                     scope='tconv2', bn_decay=bn_decay)
    #net = tf_util.conv2d(net, 1024, [1,1],
    #                     padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training,
    #                     scope='tconv3', bn_decay=bn_decay)
    maxpool_net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')
    avg_pool2d_net = tf_util.avg_pool2d(net, [num_point,1],
                             padding='VALID', scope='tavgpool')
    
    net = tf.concat([avg_pool2d_net, maxpool_net], axis=3)
    
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==9)
        weights = tf.get_variable('weights', [256, 9*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [9*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 9, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    #net = tf_util.conv2d(net, 128, [1,1],
    #                     padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training,
    #                     scope='tconv2', bn_decay=bn_decay)
    #net = tf_util.conv2d(net, 1024, [1,1],
    #                     padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training,
    #                     scope='tconv3', bn_decay=bn_decay)
    maxpool_net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='ttmaxpool')
    avg_pool2d_net = tf_util.avg_pool2d(net, [num_point,1],
                             padding='VALID', scope='ttavgpool')
    
    net = tf.concat([avg_pool2d_net, maxpool_net], axis=3)

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
