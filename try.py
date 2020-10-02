# import cv2
# import only_vgg
# import numpy as np
# import os
#
# TRAIN_DATA = './data/my_data/train/'
#
#
# def prepare_train_data(phase='train'):
#     all_train_names = [pic for pic in os.listdir(TRAIN_DATA)]
#     batch_idx = np.random.randint(0, len(all_train_names), 64)
#     print('batch idx', batch_idx)
#     bat_imgs, bat_labels = only_vgg.load_img_labels_by_name(batch_idx, phase=phase)
#
#     return bat_imgs, bat_labels
#
#
# train_images, train_labels = prepare_train_data(phase='train')
#
# print(train_images[0].max())
import tensorflow as tf
import numpy
import os
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
train_labels_pl = [[1,2,0,0,0,0], [11,220,0,0], [111,222,333,0,0,0], [1111,2222,3333,0,0], [6,7,8,9,0,0], [8,0,0,0,0,0]]
spa_train_labels = ctc_label_dense_to_sparse(labels=train_labels_pl, label_lengths=tf.constant(
    [6 for _ in range(6)]))
sess = tf.Session()
print(sess.run(spa_train_labels))
