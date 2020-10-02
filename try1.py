import tensorflow as tf
from tensorflow.contrib.layers import dense_to_sparse
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse

#

y_true = tf.constant([[0, 23, 43, 0, 10], [3,5,2,5,10]], dtype=tf.int32)  # 长度为1会报错？？不能转sparse
y_true_len = tf.constant([20, 10], dtype=tf.int32)  # shape(1,)
y_true_sparse = ctc_label_dense_to_sparse(labels=y_true, label_lengths=y_true_len)  # dense_shape(1, 3)
# # y_true_2 = dense_to_sparse(y_true, )  # dense_shape(1, 3)
#
# # y_pred = tf.constant([[[0, 8, 4, 9, 6]], [[0, -2, -4, 9, 5]], [[2, 3, 4, 5, 6]]], dtype=tf.float32)  # shape(3, 1, 5)
# # y_pred_len = tf.constant([2], dtype=tf.int32)  # shape(1, )
# # ctcloss = tf.nn.ctc_loss(labels=tf.cast(y_true_sparse, tf.int32), inputs=y_pred, sequence_length=y_pred_len)
# #
with tf.Session() as sess:
#     print(sess.run(ctcloss))
# #     print(sess.run(y_true_sparse.dense_shape))
    print(sess.run(y_true_sparse))
    # print(sess.run(y_true_2))
#     print(y_pred.shape)
#     print(y_true.shape)

# import matplotlib.pyplot as plt
#
#
# def add(a, b):
#     return a+b
#
#
# print('start plting')
# plt.plot([1,2,3], [4,5,67])
# print('start plting')
# plt.show()

