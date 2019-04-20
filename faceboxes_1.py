import tensorflow as tf
import numpy as np
import os
import cv2
from network import FaceBox
import anchors
import pickle
import data
import multiprocessing
import augmenter
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def count_number_trainable_params(scope = ""):
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    vars_chk = None
    if scope == "": 
        vars_chk = tf.trainable_variables()
    else: 
        vars_chk = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    for trainable_variable in vars_chk:
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params


def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shape.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params


def frozen_graph_to_tflite():
    graph_def_file = "./models/faceboxes.pb"
    input_arrays = ["inputs"]
    output_arrays = ['out_locs', 'out_confs']

    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open("./models/faceboxes.tflite", "wb").write(tflite_model)


def save_tflite(sess, save_path, save_name='faceboxes.tflite'):
    converter = tf.lite.TFLiteConverter.from_session(sess, ['inputs'], ['out_locs', 'out_confs'])
    tflite_model = converter.convert()
    open(os.path.join(save_path, save_name), "wb").write(tflite_model)


def save_pbtxt(sess, save_path, save_name='graph.pbtxt', output_node_names=['inputs', 'out_locs', 'out_confs']):
    print('save model graph to .pbtxt: %s' % os.path.join(save_path, save_name))
    save_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
    tf.train.write_graph(save_graph, '', os.path.join(save_path, save_name))


# 保存为pb格式
def save_pb(sess, save_path, save_name='faceboxes.pb', output_node_names=['inputs', 'out_locs', 'out_confs']):
    print('save model to .pb: %s' % os.path.join(save_path, save_name))
    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    # 此处务必和前面的输入输出对应上，其他的不用管
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

    with tf.gfile.FastGFile(os.path.join(save_path, save_name), mode='wb') as f:
        f.write(constant_graph.SerializeToString())


# 加载pb格式
def load_pb(sess, load_path, save_name='faceboxes.pb'):
    # sess = tf.Session()
    with gfile.FastGFile(os.path.join(load_path, save_name), mode='rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # # 需要有一个初始化的过程
    # sess.run(tf.global_variables_initializer())
    # # 需要先复原变量
    # print(sess.run('b:0'))
    # # 下面三句，是能否复现模型的关键
    # # 输入
    # input_x = sess.graph.get_tensor_by_name('x:0')  # 此处的x一定要和之前保存时输入的名称一致！
    # input_y = sess.graph.get_tensor_by_name('y:0')  # 此处的y一定要和之前保存时输入的名称一致！
    # op = sess.graph.get_tensor_by_name('op_to_store:0')  # 此处的op_to_store一定要和之前保存时输出的名称一致！
    # ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
    # print(ret)


def train_iter(anchors_vec, loss, imgs, lbls, normalised=False):
    locs, confs = anchors.encode_batch(anchors_vec, lbls, threshold=0.35)
    feed_dict = {
        inputs: imgs,
        training: True,
        target_locs: locs,
        target_confs: confs
    }

    pred_confs, pred_locs,  _, loss, iter = sess.run(
        fetches=[p_confs, out_locs, train_op, loss, global_iter_val],
        feed_dict=feed_dict)

    pred_boxes = anchors.decode_batch(anchors_vec, pred_locs, pred_confs)
    mAP = anchors.compute_mAP(imgs, lbls, pred_boxes, normalised=normalised)
    print('Iter:', iter[0], 'LR:', iter[1], 'Loss:', loss, 'mAP:', mAP, 'Max:', np.max(pred_locs), 'Min:', np.min(pred_locs))
    return pred_confs, pred_locs, loss, summary, mAP


# 测试迭代器
def test_iter(imgs):
    feed_dict = {
        inputs: imgs,
        training: False
    }
    pred_confs, pred_locs = sess.run([p_confs, out_locs], feed_dict=feed_dict)
    return pred_confs, pred_locs

# def save_ckpt(save_path, save_name):


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    np.set_printoptions(suppress=True)
    data_train_source = './wider_train.p'
    data_test_source = './wider_test.p'
    data_train_dir = '../Data/WIDER/WIDER_train/images/'
    data_test_dir = '../Data/WIDER/WIDER_val/images/'
    save_path = './models/'
    model_name = 'faceboxes.ckpt'
    PRINT_FREQ = 500
    TEST_FREQ = 20
    SAVE_FREQ = 1000
    BATCH_SIZE = 32
    IM_S = 1024
    IM_CHANNELS = 3
    N_WORKERS = 20
    MAX_PREBUFF_LIM = 20
    IOU_THRESH = 0.5
    USE_NORM = True
    CONFIG = [[1024, 1024, 32, 32, 32, 32, 4], 
            [1024, 1024, 32, 32, 64, 64, 2],
            [1024, 1024, 32, 32, 128, 128, 1],
            [1024, 1024, 64, 64, 256, 256, 1],
            [1024, 1024, 128, 128, 512, 512, 1]]
    IS_AUG = True
    USE_MP = False
    USE_AUG_TF = True
    if USE_AUG_TF and USE_MP:
        raise ValueError("Can't use TF augmenter with multiprocessing")
    # NOTE: SSD variances are set in the anchors.py file
    boxes_vec, boxes_lst, stubs = anchors.get_boxes(CONFIG, normalised=USE_NORM)
    tf.reset_default_graph()

    train_data = pickle.load(file=open(data_train_source, 'rb'))
    test_data = pickle.load(file=open(data_test_source, 'rb'))

    svc_train = None
    if IS_AUG and not USE_AUG_TF:
        aug_params = {'use_tf': False}
        if USE_MP:
            mp_dict = {'lim': MAX_PREBUFF_LIM, 'n': N_WORKERS, 'b_s': BATCH_SIZE}
            svc_train = data.DataService(train_data, aug_params, data_train_dir, (IM_S, IM_S), mp_dict, normalised=USE_NORM)
        else:
            svc_train = data.DataService(train_data, aug_params, data_train_dir, (IM_S, IM_S), None, normalised=USE_NORM)
        print('Starting augmenter...')
        svc_train.start()
        print('Running model...')
    else:
        svc_train = data.DataService(train_data, False, data_train_dir, (IM_S, IM_S), normalised=USE_NORM)
    svc_test = data.DataService(test_data, False, data_test_dir, (IM_S, IM_S), normalised=USE_NORM)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if IS_AUG and USE_AUG_TF:
            gpu_aug = augmenter.AugmenterGPU(sess, (IM_S, IM_S))
            aug_params = {'use_tf': True, 'augmenter': gpu_aug}
            if USE_MP:
                mp_dict = {'lim': MAX_PREBUFF_LIM, 'n': N_WORKERS, 'b_s': BATCH_SIZE}
                svc_train = data.DataService(train_data, aug_params, data_train_dir, (IM_S, IM_S), mp_dict, normalised=USE_NORM)
            else:
                svc_train = data.DataService(train_data, aug_params, data_train_dir, (IM_S, IM_S), None, normalised=USE_NORM)
        print('Building model...')
        # fb_model = FaceBox(sess, (BATCH_SIZE, IM_S, IM_S, IM_CHANNELS), boxes_vec, normalised=USE_NORM)
        fb_model = FaceBox((BATCH_SIZE, IM_S, IM_S, IM_CHANNELS), boxes_vec, normalised=USE_NORM)

        training = tf.placeholder(tf.bool, shape=[], name='training')
        out_locs, out_confs, p_confs = fb_model(training)

        inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IM_S, IM_S, IM_CHANNELS), name="inputs")
        target_locs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, boxes_vec, 4), name='target_locs')
        target_confs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, boxes_vec, 1), name='target_confs')

        loss = fb_model.compute_loss(out_locs, out_confs, target_locs, target_confs)
        loss += tf.losses.get_regularization_loss()  # Add regularisation
        # tf.summary.scalar('Loss', loss)

        # ==================================================================================
        # lr
        global_step = tf.Variable(0, trainable=False)
        i_plus = tf.assign(global_step, global_step + 1)
        boundaries = [200000.0, 400000.0]
        values = [0.001, 0.0001, 0.00001]
        lr = tf.train.piecewise_constant(tf.to_float(global_step), boundaries, values, name='lr_select')
        global_iter_val = [global_step, lr]

        # # ==================================================================================
        # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(self.extra_update_ops):
        #     self.train = tf.train.AdamOptimizer(self.lr, epsilon=0.1).minimize(self.loss)
        #     self.train = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True).minimize(self.loss)

        # ==================================================================================
        # TODO
        optimizer = tf.train.AdamOptimizer(lr, epsilon=0.1)
        minimize_op = optimizer.minimize(loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        # self.merged = tf.summary.merge_all()

        # ==================================================================================
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
        saver = tf.train.Saver()
        last_ckpt = tf.train.latest_checkpoint(save_path)

        if last_ckpt is not None:
            saver.restore(sess, last_ckpt)
            print('Succesfully loaded saved model')
        else:
            print('Model not found - using default initialisation!')
            sess.run(tf.global_variables_initializer())

        # ==================================================================================
        i = 0
        train_mAP_pred = []
        train_loss = []
        test_mAP_pred = []
        while True:
            print(' Iteration ', i, '                                                ', end='\r')
            i += 1
            imgs, lbls = None, None
            if USE_MP: 
                imgs, lbls = svc_train.pop()
            else:
                imgs, lbls = svc_train.random_sample(BATCH_SIZE)


            pred_confs, pred_locs, loss, summary, mAP = train_iter(boxes_vec, loss, imgs, lbls)
            train_loss.append(loss)
            train_mAP_pred.append(mAP)

            if i % PRINT_FREQ == 0:
                print("")
                print('Iteration: ', i)
                print('Mean train loss: ', np.mean(train_loss))
                print('Mean train mAP: ', np.mean(train_mAP_pred))
                train_mAP_pred = []
                train_loss = []
            if i % TEST_FREQ == 0:
                for j in range(25):
                    imgs, lbls = svc_test.random_sample(BATCH_SIZE)
                    pred_confs, pred_locs = test_iter(imgs)
                    pred_boxes = anchors.decode_batch(boxes_vec, pred_locs, pred_confs)
                    test_mAP_pred.append(anchors.compute_mAP(imgs, lbls, pred_boxes, normalised=USE_NORM))
                print('Mean test mAP: ', np.mean(test_mAP_pred))
                test_mAP_pred = []
            # if i % SAVE_FREQ == 0:
                print('Saving model...')
                saver.save(sess, os.path.join(save_path, model_name), global_step=i)
                save_pb(sess, save_path)
                save_pbtxt(sess, save_path)


