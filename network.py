import tensorflow as tf
import numpy as np
import anchors

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class FaceBox(object):
    def __init__(self, input_shape, anchors_in, normalised=False, anchors_scale=anchors.SCALE_FACTOR):
        # self.sess = sess
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.base_init = tf.contrib.layers.xavier_initializer()             # Initialise weights
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.0005)      # Initialise regularisation
        self.anchor_len = anchors_in.shape[0]
        self.anchors_bbox = tf.to_float(tf.constant(anchors_in))
        self.anchors_bbox_scale = anchors_scale
        self.normalised = normalised
        # self.build_graph()

    def batch_norm(self, inputs, training, name, axis=3):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        # return tf.compat.v1.layers.batch_normalization(
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=training,
            fused=True,
            name=name+"_bn")

    def CReLU(self, in_x, training, name):
        with tf.variable_scope(name):
            # x = tf.layers.batch_normalization(in_x, training=self.is_training, name=name + '_batch')
            x = self.batch_norm(in_x, training, name)
            return tf.nn.crelu(x, name=name + '_crelu')

    def Inception(self, in_x, training, name):
        with tf.variable_scope(name):
            path_1 = tf.layers.conv2d(in_x, 32,
                                      kernel_size=[1, 1],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_1_1',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            # path_1 = tf.layers.batch_normalization(path_1, training=self.is_training, name=name + 'path_1_batch')
            path_1 = self.batch_norm(path_1, training, name + 'path_1')

            path_2 = tf.layers.max_pooling2d(in_x, [3, 3], 1, name=name + 'pool_1_2', padding='SAME')
            path_2 = tf.layers.conv2d(path_2, 32,
                                      kernel_size=[1, 1],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_1_2',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            # path_2 = tf.layers.batch_normalization(path_2, training=self.is_training, name=name + 'path_2_batch')
            path_2 = self.batch_norm(path_2, training, name + 'path_2')

            path_3 = tf.layers.conv2d(in_x, 24,
                                      kernel_size=[1, 1],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_1_3',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            path_3 = tf.layers.conv2d(path_3, 32,
                                      kernel_size=[3, 3],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_2_3',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            # path_3 = tf.layers.batch_normalization(path_3, training=self.is_training, name=name + 'path_3_batch')
            path_3 = self.batch_norm(path_3, training, name + 'path_3')

            path_4 = tf.layers.conv2d(in_x, 24,
                                      kernel_size=[1, 1],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_1_4',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            path_4 = tf.layers.conv2d(path_4, 32,
                                      kernel_size=[3, 3],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_2_4',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            path_4 = tf.layers.conv2d(path_4, 32,
                                      kernel_size=[3, 3],
                                      strides=1,
                                      kernel_initializer=self.base_init,
                                      kernel_regularizer=self.reg_init,
                                      name=name + 'conv_3_4',
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
            # path_4 = tf.layers.batch_normalization(path_4, training=self.is_training, name=name + 'path_4_batch')
            path_4 = self.batch_norm(path_4, training, name + 'path_4')

            return tf.concat([path_1, path_2, path_3, path_4], axis=-1)

    def build_anchor(self, in_x, num_out, name):
        with tf.variable_scope(name):
            bbox_loc_conv = tf.layers.conv2d(in_x, num_out*4,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             kernel_initializer=self.base_init,
                                             kernel_regularizer=self.reg_init,
                                             activation=None,
                                             name=name + '_anchor_loc_conv',
                                             padding='SAME')
            bbox_class_conv = tf.layers.conv2d(in_x, num_out*2,
                                               kernel_size=[3, 3],
                                               strides=1,
                                               kernel_initializer=self.base_init,
                                               kernel_regularizer=self.reg_init,
                                               activation=None,
                                               name=name + '_anchor_conf_conv',
                                               padding='SAME')
            return bbox_loc_conv, bbox_class_conv

    def hard_negative_mining(self, conf_loss, pos_ids, mult=3, min_negs=10):
        with tf.name_scope('hard_negative_mining') as scope:
            pos_ids = tf.unstack(pos_ids)
            neg_ids = [tf.logical_not(p) for p in pos_ids]
            conf_loss = tf.unstack(conf_loss)
            loss_out = []
            for c_i in range(self.batch_size):
                c_pos_ids = pos_ids[c_i]
                c_neg_ids = neg_ids[c_i]
                c_num_neg = tf.cast(tf.reduce_sum(tf.cast(c_neg_ids, tf.float32)), tf.int32)
                c_num_pos = tf.cast(tf.reduce_sum(tf.cast(c_pos_ids, tf.float32)), tf.int32)
                c_conf_loss = conf_loss[c_i]
                # c_l1_loss = l1_loss[c_i]
                # Extract negative confidence losses only
                loss_conf_neg = tf.reshape(tf.boolean_mask(c_conf_loss, c_neg_ids), [c_num_neg])
                loss_conf_pos = tf.reshape(tf.boolean_mask(c_conf_loss, c_pos_ids), [c_num_pos])
                # loss_l1_pos = tf.reshape(tf.boolean_mask(c_l1_loss, c_pos_ids), [c_num_pos])
                c_neg_cap = tf.cast(mult * c_num_pos, tf.int32)
                c_neg_cap = tf.maximum(min_negs, c_neg_cap) # Cap minimum negative value to min_negs
                c_neg_cap = tf.minimum(c_neg_cap, c_num_neg) # Cap minimum values to max # = anchor_len
                loss_conf_k_neg, _ = tf.nn.top_k(loss_conf_neg, k=c_neg_cap, sorted=True)
                loss_out.append(tf.concat((loss_conf_pos, loss_conf_k_neg), axis=0))
            return tf.concat(loss_out, axis=0)

    def compute_loss(self, loc_preds, conf_preds, loc_true, conf_true):
        with tf.name_scope('loss') as scope:
            loc_preds = tf.reshape(loc_preds, (self.batch_size, -1, 4))
            conf_preds = tf.reshape(conf_preds, (self.batch_size, -1, 2))
            loc_true = tf.reshape(loc_true, (self.batch_size, -1, 4))
            conf_true = tf.cast(tf.reshape(conf_true, (self.batch_size, -1)), tf.int32)
            conf_true_oh = tf.one_hot(conf_true, 2)

            positive_check = tf.reshape(tf.cast(tf.equal(conf_true, 1), tf.float32), (self.batch_size, self.anchor_len))
            pos_ids = tf.cast(positive_check, tf.bool)
            n_pos = tf.maximum(tf.reduce_sum(positive_check), 1)

            # Smoothed L1 loss
            l1_loss = tf.losses.huber_loss(loc_preds, loc_true, reduction=tf.losses.Reduction.NONE)
            # Zero out L1 loss for negative boxes
            l1_loss = positive_check * tf.reduce_sum(l1_loss, axis=-1)

            # conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels =
            # tf.squeeze(tf.to_int32(conf_true)), logits = conf_preds)
            conf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=conf_true_oh, logits=conf_preds)
            conf_loss = self.hard_negative_mining(conf_loss, pos_ids)

            loss = (tf.reduce_sum(l1_loss) + tf.reduce_sum(conf_loss))/n_pos
            return loss

    def __call__(self, inputs, training):
        # Process inputs
        self.inputs = tf.placeholder(tf.float32,
                                     shape=(self.batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3]),
                                     name="inputs")
        self.inputs = self.inputs / 255.0  # Normalise to 0-1
        self.inputs = 2.0 * self.inputs - 1.0  # Makes the same as InceptionNet
        # self.is_training = tf.placeholder(tf.bool, name="is_training")
        global_step = tf.Variable(0, trainable=False)
        self.i_plus = tf.assign(global_step, global_step + 1)

        boundaries = [200000.0, 400000.0]
        values = [0.001, 0.0001, 0.00001]
        self.lr = tf.train.piecewise_constant(tf.to_float(global_step), boundaries, values, name='lr_select')
        self.global_iter_val = [global_step, self.lr]

        bbox_locs = []
        bbox_confs = []

        # print('Building RDCL...')
        conv_1 = tf.layers.conv2d(self.inputs, 24,
                                  kernel_size=[7, 7],
                                  strides=4,
                                  kernel_initializer=self.base_init,
                                  kernel_regularizer=self.reg_init,
                                  name='Conv1',
                                  padding='SAME')
        conv_1_crelu = self.CReLU(conv_1, training, 'CReLU_1')
        conv_1_pool = tf.layers.max_pooling2d(conv_1_crelu, [3, 3], 2, name='Pool1', padding='SAME')
        # conv_1_pool = tf.layers.batch_normalization(conv_1_pool, training=self.is_training, name='conv_1_pool_batch')
        conv_1_pool = self.batch_norm(conv_1_pool, training, 'conv_1_pool')

        conv_2 = tf.layers.conv2d(conv_1_pool, 64,
                                  kernel_size=[5, 5],
                                  strides=2,
                                  kernel_initializer=self.base_init,
                                  kernel_regularizer=self.reg_init,
                                  name='Conv2',
                                  padding='SAME')
        conv_2_crelu = self.CReLU(conv_2, training, 'CReLU_2')
        conv_2_pool = tf.layers.max_pooling2d(conv_2_crelu, [3, 3], 2, name='Pool2', padding='SAME')
        # conv_2_pool = tf.layers.batch_normalization(conv_2_pool, training=self.is_training, name='conv_2_pool_batch')
        conv_2_pool = self.batch_norm(conv_2_pool, training, 'conv_2_pool')

        # print('Building Inception...')
        incept_1 = self.Inception(conv_2_pool, training, 'inception_1')
        incept_2 = self.Inception(incept_1, training, 'inception_2')
        incept_3 = self.Inception(incept_2, training, 'inception_3')

        l, c = self.build_anchor(incept_3, 21, 'anchor_incept_3')
        bbox_locs.append(l)
        bbox_confs.append(c)

        conv_3_1 = tf.layers.conv2d(incept_3, 128,
                                    kernel_size=[1, 1],
                                    strides=1,
                                    kernel_initializer=self.base_init,
                                    kernel_regularizer=self.reg_init,
                                    activation=tf.nn.leaky_relu,
                                    name='Conv3_1',
                                    padding='SAME')

        conv_3_2 = tf.layers.conv2d(conv_3_1, 256,
                                    kernel_size=[3, 3],
                                    strides=2,
                                    kernel_initializer=self.base_init,
                                    kernel_regularizer=self.reg_init,
                                    activation=tf.nn.leaky_relu,
                                    name='Conv3_2',
                                    padding='SAME')
        # conv_3_2 = tf.layers.batch_normalization(conv_3_2, training=self.is_training, name='conv_3_2_batch')
        conv_3_2 = self.batch_norm(conv_3_2, training, 'conv_3_2')

        l, c = self.build_anchor(conv_3_2, 1, 'anchor_conv_3_2')
        bbox_locs.append(l)
        bbox_confs.append(c)

        conv_4_1 = tf.layers.conv2d(conv_3_2, 128,
                                    kernel_size=[1, 1],
                                    strides=1,
                                    kernel_initializer=self.base_init,
                                    kernel_regularizer=self.reg_init,
                                    activation=tf.nn.leaky_relu,
                                    name='Conv4_1',
                                    padding='SAME')

        # conv_4_1 = tf.layers.batch_normalization(conv_4_1, training=self.is_training, name='conv_4_1_batch')
        conv_4_1 = self.batch_norm(conv_4_1, training, 'conv_4_1')
        conv_4_2 = tf.layers.conv2d(conv_4_1, 256,
                                    kernel_size=[3, 3],
                                    strides=2,
                                    kernel_initializer=self.base_init,
                                    kernel_regularizer=self.reg_init,
                                    activation=tf.nn.leaky_relu,
                                    name='Conv4_2',
                                    padding='SAME')

        l, c = self.build_anchor(conv_4_2, 1, 'anchor_conv_4_2')
        bbox_locs.append(l)
        bbox_confs.append(c)

        self.out_locs = tf.concat([tf.reshape(i, [self.batch_size, -1, 4]) for i in bbox_locs], axis=-2)
        self.out_confs = tf.concat([tf.reshape(i, [self.batch_size, -1, 2]) for i in bbox_confs], axis=-2)
        self.out_locs = tf.reshape(self.out_locs, [self.batch_size, self.anchor_len, 4], name='out_locs')
        # print('Locs min cap: ', np.log(1/512)*self.anchors_bbox_scale[1])
        # print('Locs max cap: ', np.log(1024/32)*self.anchors_bbox_scale[1])

        self.out_confs = tf.reshape(self.out_confs, [self.batch_size, self.anchor_len, 2], name='out_confs')
        self.p_confs = tf.nn.softmax(self.out_confs, name='p_confs')

        print('Output loc shapes', self.out_locs.get_shape())
        print('Output conf shapes', self.out_confs.get_shape())

        self.target_locs = tf.placeholder(tf.float32, shape=(self.batch_size, self.anchor_len, 4), name='target_locs')
        self.target_confs = tf.placeholder(tf.float32, shape=(self.batch_size, self.anchor_len, 1), name='target_confs')

        self.loss = self.compute_loss(self.out_locs, self.out_confs, self.target_locs, self.target_confs)
        self.loss += tf.losses.get_regularization_loss()  # Add regularisation
        tf.summary.scalar('Loss', self.loss)

        # ==================================================================================
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train = tf.train.AdamOptimizer(self.lr, epsilon=0.1).minimize(self.loss)
            self.train = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True).minimize(self.loss)

        # ==================================================================================
        # TODO
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.1)
        minimize_op = optimizer.minimize(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group(minimize_op, update_ops)

        self.merged = tf.summary.merge_all()