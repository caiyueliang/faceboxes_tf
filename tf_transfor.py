import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def frozen_graph_to_tflite():
    graph_def_file = "./models/faceboxes.pb"
    input_arrays = ["inputs"]
    output_arrays = ['out_locs', 'out_confs']

    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open("./models/faceboxes.tflite", "wb").write(tflite_model)


def save_tflite():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        converter = tf.lite.TFLiteConverter.from_session(sess, ['inputs'], ['out_locs', 'out_confs'])
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)


def save_pbtxt(save_path, save_name='graph.pbtxt', output_node_names=['inputs', 'out_locs', 'out_confs']):
    with tf.Session() as sess:
        print('save model graph to .pbtxt: %s' % os.path.join(save_path, save_name))
        save_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
        tf.train.write_graph(save_graph, '', os.path.join(save_path, save_name))


# 保存为pb格式
def save_pb(save_path, save_name='faceboxes.pb', output_node_names=['inputs', 'out_locs', 'out_confs']):
    with tf.Session() as sess:
        print('save model to .pb: %s' % os.path.join(save_path, save_name))
        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        # 此处务必和前面的输入输出对应上，其他的不用管
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

        with tf.gfile.FastGFile(os.path.join(save_path, save_name), mode='wb') as f:
            f.write(constant_graph.SerializeToString())


# 加载pb格式
def load_pb(load_path, save_name='faceboxes.pb'):
    # sess = tf.Session()
    with tf.Session() as sess:
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
