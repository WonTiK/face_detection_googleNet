# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
from collections import defaultdict
from tensorflow.python.ops import init_ops

class DummyScope(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class GPUNetworkBuilder(object):
    """This class provides convenient methods for constructing feed-forward
    networks with internal data layout of 'NCHW'.
    """

    def __init__(self,
                 is_training,
                 dtype=tf.float32,
                 activation='RELU',
                 use_batch_norm=True,
                 batch_norm_config={'decay': 0.9,
                                    'epsilon': 1e-4,
                                    'scale': True,
                                    'zero_debias_moving_mean': False},
                 use_xla=False):
        self.dtype = dtype
        self.activation_func = activation
        self.is_training = is_training
        self.use_batch_norm = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self._layer_counts = defaultdict(lambda: 0)
        if use_xla:
            self.jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        else:
            self.jit_scope = DummyScope

    def _count_layer(self, layer_type):
        idx = self._layer_counts[layer_type]
        name = layer_type + str(idx)
        self._layer_counts[layer_type] += 1
        return name

    def _get_variable(self, name, shape, dtype=None,
                      initializer=None, seed=None):
        if dtype is None:
            dtype = self.dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        elif (isinstance(initializer, float) or
              isinstance(initializer, int)):
            initializer = tf.constant_initializer(float(initializer))
        return tf.get_variable(name, shape, dtype, initializer)

    def _to_nhwc(self, x):
        return tf.transpose(x, [0, 2, 3, 1])

    def _from_nhwc(self, x):
        return tf.transpose(x, [0, 3, 1, 2])

    def _bias(self, input_layer):
        num_outputs = input_layer.get_shape().as_list()[1]
        biases = self._get_variable('biases', [num_outputs], input_layer.dtype,
                                    initializer=0)
        if len(input_layer.get_shape()) == 4:
            return tf.nn.bias_add(input_layer, biases,
                                  data_format='NCHW')
        else:
            return input_layer + biases

    def _batch_norm(self, input_layer, scope):
        return tf.contrib.layers.batch_norm(input_layer,
                                            is_training=self.is_training,
                                            scope=scope,
                                            data_format='NCHW',
                                            fused=True,
                                            **self.batch_norm_config)

    def _bias_or_batch_norm(self, input_layer, scope, use_batch_norm):
        if use_batch_norm is None:
            use_batch_norm = self.use_batch_norm
        if use_batch_norm:
            return self._batch_norm(input_layer, scope)
        else:
            return self._bias(input_layer)

    def input_layer(self, input_layer):
        """Converts input data into the internal format"""
        with self.jit_scope():
            x = self._from_nhwc(input_layer)
            x = tf.cast(x, self.dtype)
            # Rescale and shift to [-1,1]
            x = x * (1. / 127.5) - 1
        return x

    def conv(self, input_layer, num_filters, filter_size,
             filter_strides=(1, 1), padding='SAME',
             activation=None, use_batch_norm=None):
        """Applies a 2D convolution layer that includes bias or batch-norm
        and an activation function.
        """
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_inputs, num_filters]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('conv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            if padding == 'SAME_RESNET':  # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                padding = [[0, 0], [0, 0],
                           [pad_beg, pad_end], [pad_beg, pad_end]]
                input_layer = tf.pad(input_layer, padding)
                padding = 'VALID'
            x = tf.nn.conv2d(input_layer, kernel, strides,
                             padding=padding, data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x

    def deconv(self, input_layer, num_filters, filter_size,
               filter_strides=(2, 2), padding='SAME',
               activation=None, use_batch_norm=None):
        """Applies a 'transposed convolution' layer that includes bias or
        batch-norm and an activation function.
        """
        num_inputs = input_layer.get_shape().as_list()[1]
        ih, iw = input_layer.get_shape().as_list()[2:]
        output_shape = [-1, num_filters,
                        ih * filter_strides[0], iw * filter_strides[1]]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_filters, num_inputs]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('deconv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            x = tf.nn.conv2d_transpose(input_layer, kernel, output_shape,
                                       strides, padding=padding,
                                       data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x

    def activate(self, input_layer, funcname=None):
        """Applies an activation function"""
        if isinstance(funcname, tuple):
            funcname = funcname[0]
            params = funcname[1:]
        if funcname is None:
            funcname = self.activation_func
        if funcname == 'LINEAR':
            return input_layer
        activation_map = {
            'RELU': tf.nn.relu,
            'RELU6': tf.nn.relu6,
            'ELU': tf.nn.elu,
            'SIGMOID': tf.nn.sigmoid,
            'TANH': tf.nn.tanh,
            'LRELU': lambda x, name: tf.maximum(params[0] * x, x, name=name)
        }
        return activation_map[funcname](input_layer, name=funcname.lower())

    def pool(self, input_layer, funcname, window_size,
             window_strides=(2, 2),
             padding='VALID'):
        """Applies spatial pooling"""
        pool_map = {
            'MAX': tf.nn.max_pool,
            'AVG': tf.nn.avg_pool
        }
        kernel_size = [1, 1, window_size[0], window_size[1]]
        kernel_strides = [1, 1, window_strides[0], window_strides[1]]
        return pool_map[funcname](input_layer, kernel_size, kernel_strides,
                                  padding, data_format='NCHW',
                                  name=funcname.lower())

    def project(self, input_layer, num_outputs, height, width,
                activation=None):
        """Linearly projects to an image-like tensor"""
        with tf.variable_scope(self._count_layer('project')):
            x = self.fully_connected(input_layer, num_outputs * height * width,
                                     activation=activation)
            x = tf.reshape(x, [-1, num_outputs, height, width])
            return x

    def flatten(self, input_layer):
        """Flattens the spatial and channel dims into a single dim (4D->2D)"""
        # Note: This ensures the output order matches that of NHWC networks
        input_layer = self._to_nhwc(input_layer)
        input_shape = input_layer.get_shape().as_list()
        num_inputs = input_shape[1] * input_shape[2] * input_shape[3]
        return tf.reshape(input_layer, [-1, num_inputs], name='flatten')

    def spatial_avg(self, input_layer):
        """Averages over spatial dimensions (4D->2D)"""
        return tf.reduce_mean(input_layer, [2, 3], name='spatial_avg')

    def fully_connected(self, input_layer, num_outputs, activation=None):
        """Applies a fully-connected set of weights"""
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_size = [num_inputs, num_outputs]
        with tf.variable_scope(self._count_layer('fully_connected')):
            kernel = self._get_variable('weights', kernel_size,
                                        input_layer.dtype)
            x = tf.matmul(input_layer, kernel)
            x = self._bias(x)
            x = self.activate(x, activation)
            return x

    def inception_module(self, input_layer, name, cols):
        """Applies an inception module with a given form"""
        with tf.name_scope(name):
            col_layers = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                x = input_layer
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    if ltype == 'conv':
                        x = self.conv(x, *args)
                    elif ltype == 'pool':
                        x = self.pool(x, *args)
                    elif ltype == 'share':
                        # Share matching layer from previous column
                        x = col_layers[c - 1][l]
                    else:
                        raise KeyError("Invalid layer type for " +
                                       "inception module: '%s'" % ltype)
                    col_layers[c].append(x)
            catdim = 1
            catvals = [layers[-1] for layers in col_layers]
            x = tf.concat(catvals, catdim)
            return x

    def residual(self, input_layer, net, scale=1.0, activation='RELU'):
        """Applies a residual layer"""
        input_size = input_layer.get_shape().as_list()
        num_inputs = input_size[1]
        output_layer = scale * net(self, input_layer)
        output_size = output_layer.get_shape().as_list()
        num_outputs = output_size[1]
        kernel_strides = (input_size[2] // output_size[2],
                          input_size[3] // output_size[3])
        with tf.name_scope('residual'):
            if (num_outputs != num_inputs or
                    kernel_strides[0] != 1 or
                    kernel_strides[1] != 1):
                input_layer = self.conv(input_layer, num_outputs, [1, 1],
                                        kernel_strides, activation='LINEAR')
            with self.jit_scope():
                x = self.activate(input_layer + output_layer, activation)
            return x

    def dropout(self, input_layer, keep_prob=0.5):
        """Applies a dropout layer if is_training"""
        if self.is_training:
            dtype = input_layer.dtype
            with tf.variable_scope(self._count_layer('dropout')):
                keep_prob_tensor = tf.constant(keep_prob, dtype=dtype)
                return tf.nn.dropout(input_layer, keep_prob_tensor)
        else:
            return input_layer

def inference_googlenet(net, input_layer):
    """GoogLeNet model
    https://arxiv.org/abs/1409.4842
    """
    net.use_batch_norm = False

    def inception_v1(net, x, k, l, m, n, p, q):
        cols = [[('conv', k, (1, 1))],
                [('conv', l, (1, 1)), ('conv', m, (3, 3))],
                [('conv', n, (1, 1)), ('conv', p, (5, 5))],
                [('pool', 'MAX', (3, 3), (1, 1), 'SAME'), ('conv', q, (1, 1))]]
        return net.inception_module(x, 'incept_v1', cols)

    print('input_layer=', input_layer)
    x = net.input_layer(input_layer)
    print('x=', x)
    x = net.conv(x, 64, (7, 7), (2, 2))
    print('x=', x)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    print('x=', x)
    x = net.conv(x, 64, (1, 1))
    print('x=', x)
    x = net.conv(x, 192, (3, 3))
    print('x=', x)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    print('x=', x)
    x = inception_v1(net, x, 64, 96, 128, 16, 32, 32)
    x = inception_v1(net, x, 128, 128, 192, 32, 96, 64)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    x = inception_v1(net, x, 192, 96, 208, 16, 48, 64)
    x = inception_v1(net, x, 160, 112, 224, 24, 64, 64)
    x = inception_v1(net, x, 128, 128, 256, 24, 64, 64)
    x = inception_v1(net, x, 112, 144, 288, 32, 64, 64)
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = inception_v1(net, x, 384, 192, 384, 48, 128, 128)
    x = net.spatial_avg(x)
    return x

def eval_func(images, var_scope):
    net = GPUNetworkBuilder(
        False, dtype=tf.float32, use_xla=False)
    model_func = inference_googlenet
    output = model_func(net, images)
    logits_g = net.fully_connected(output, 2, activation='LINEAR')
    logits_a = net.fully_connected(output, 8, activation='LINEAR')
    if logits_g.dtype != tf.float32:
        logits_g = tf.cast(logits_g, tf.float32)
    if logits_a.dtype != tf.float32:
        logits_a = tf.cast(logits_a, tf.float32)
    with tf.device('/cpu:0'):
        logits_g = tf.nn.softmax(logits_g)
        logits_a = tf.nn.softmax(logits_a)

    return logits_g, logits_a

def eval_once(sess, logits_g, logits_a):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    result = sess.run([logits_g, logits_a])

    return result

def evaluate(images, checkpoint_dir):
    """Eval googleNet for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        res_images = tf.cast(tf.image.resize_images(images, [224, 224]), dtype=tf.float32)
        res_images = tf.reshape(res_images, [1, 224, 224, 3])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # Build inference Graph.
        #print('>>>>> input original size = ', res_images)
        #   in case of images less than or larger than 227x227
        #   images = tf.image.resize_images(images, [227,227] )
        #   print('>>>>> input resized = ',images)
        with tf.variable_scope('GPU_%i' % 0, reuse=tf.AUTO_REUSE) as var_scope, \
                tf.name_scope('tower_%i' % 0):
                    logits_g, logits_a = eval_func(res_images, var_scope)

        # Restore the moving average version of the learned variables for eval.
        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(allocator_type='BFC', allow_growth=True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            # Start the queue runners.
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            try:
                result = eval_once(sess, logits_g, logits_a)
                gender_result, age_result = None, None

                if result[0][0][0] < result[0][0][1]:
                    gender_result = 'MAN'
                else:
                    gender_result = 'WOMAN'

                max = 0

                for i in range(len(result[1][0])):
                    if result[1][0][i] > result[1][0][max]:
                        max = i

                if max == 0:
                    age_result = '0-9'
                elif max == 1:
                    age_result = '10-19'
                elif max == 2:
                    age_result = '20-29'
                elif max == 3:
                    age_result = '30-39'
                elif max == 4:
                    age_result = '40-49'
                elif max == 5:
                    age_result = '50-59'
                elif max == 6:
                    age_result = '60-69'
                elif max == 7:
                    age_result = 'over 70'
                else:
                    age_result = 'ERROR'

                return gender_result, age_result

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)