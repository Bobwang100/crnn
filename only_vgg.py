import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import vgg as vgg
from tensorflow.contrib import slim as slim
from utils.data_utils import letterbox_resize, parse_line
from utils.data_aug import resize_with_bbox
import pandas as pd
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
from tensorflow.contrib.slim.nets import resnet_v1 as resnet_v1
import cv2
from tensorflow.python.ops import math_ops
from skimage.transform import resize
import data_aug


# TRAIN_DATA = './data/my_data/train/'


# out, end_points = resnet_v2.resnet_v2_152(tfx, num_classes=2, )    #  is_training=False
# out = tf.reshape(out, (-1, 2))

def train():
    TRAIN_DATA = './data/my_data/train/'
    TEST_DATA = './data/my_data/test/'
    VAL_DATA = './data/my_data/val/'
    BATCH = 5

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tfx = tf.placeholder(tf.float32, [None, 60, 120, 3])
    tfy = tf.placeholder(tf.float32, [None, 5, 11])
    # _, end_points = vgg.vgg_16(tfx, num_classes=2)

    # _, end_points = vgg.vgg_16(tfx, num_classes=2)
    # fc8 = slim.fully_connected(end_points['vgg_16/fc7'], num_outputs=55)
    out, end_points = resnet_v2.resnet_v2_152(tfx, num_classes=55)
    # out, end_points = vgg.vgg_19(tfx, num_classes=55)
    out = tf.reshape(out, (-1, 5, 11))

    loss = tf.losses.softmax_cross_entropy(tfy[0], out[0]) + \
           tf.losses.softmax_cross_entropy(tfy[1], out[1]) + \
           tf.losses.softmax_cross_entropy(tfy[2], out[2]) + \
           tf.losses.softmax_cross_entropy(tfy[3], out[3]) + \
           tf.losses.softmax_cross_entropy(tfy[4], out[4])

    train_op = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
    # train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    config = tf.ConfigProto(allow_soft_placement=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, './models_crop/transfer_learn_105000')
    # slim.assign_from_checkpoint_fn('./data/pre_trained/resnet_v2_152.ckpt',
    #                                slim.get_trainable_variables(),
    #                                ignore_missing_vars=True)
    # saver.restore(sess, './data/pre_trained/resnet_v1_50.ckpt')
    all_train_names = [pic for pic in os.listdir(TRAIN_DATA)] + [pic for pic in os.listdir(VAL_DATA)]
    for i in range(1000000):
        batch_idx = np.random.randint(0, len(all_train_names), BATCH)
        bat_imgs, bat_labels = load_img_labels_by_name(batch_idx)
        one_hot_labels = parse_labels(bat_labels)
        # print(one_hot_labels)
        losses, _ = sess.run((loss, train_op),
                             feed_dict={tfx: bat_imgs, tfy: one_hot_labels})
        if i % 2 == 0:
            print(i, 'loss', losses)
        if i % 5000 == 0:
            saver.save(sess, './models_vgg19/transfer_learn_%d' % i)

        # print(labels,'\n', one_hot_labels)


def do_img_aug(img):
    # if np.random.uniform() > 0.3:
    #     img = data_aug.sp_noise_img(img, np.random.uniform(0.03, 0.05))
    # if np.random.uniform() > 0.3:
    #     img = data_aug.contrast_bright_img(img, 0.25, 0.5)
    img = data_aug.rotate_img(img, np.random.randint(10, 30))
    return img

def test_batch():
    TRAIN_DATA = './data/my_data/train/'
    TEST_DATA = './data/my_data/test/'
    VAL_DATA = './data/my_data/val/'
    BATCH = 5

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tfx = tf.placeholder(tf.float32, [None, 60, 120, 3])
    tfy = tf.placeholder(tf.float32, [None, 5, 11])
    # _, end_points = vgg.vgg_16(tfx, num_classes=2)

    # _, end_points = vgg.vgg_16(tfx, num_classes=2)
    # fc8 = slim.fully_connected(end_points['vgg_16/fc7'], num_outputs=55)
    out, end_points = resnet_v2.resnet_v2_152(tfx, num_classes=55)
    # out, end_points = vgg.vgg_19(tfx, num_classes=55)
    out = tf.reshape(out, (-1, 5, 11))

    loss = tf.losses.softmax_cross_entropy(tfy[0], out[0]) + \
           tf.losses.softmax_cross_entropy(tfy[1], out[1]) + \
           tf.losses.softmax_cross_entropy(tfy[2], out[2]) + \
           tf.losses.softmax_cross_entropy(tfy[3], out[3]) + \
           tf.losses.softmax_cross_entropy(tfy[4], out[4])

    train_op = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
    # train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    config = tf.ConfigProto(allow_soft_placement=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('test batch')
    saver.restore(sess, './models_crop/transfer_learn_235000')
    all_test_names = [pic for pic in os.listdir(TEST_DATA)]
    df = pd.DataFrame({"file_name": [], "file_code": []})
    test_time = int(len(all_test_names)//BATCH)
    for idx in range(test_time):
        img_bch = []
        for sub_i in range(BATCH):
            img = cv2.imread(TEST_DATA + all_test_names[idx*BATCH + sub_i])
            resize_img = resize(img, (60, 120))
            crop_img = np.asarray(resize_img, np.float32)
            crop_img = crop_img / 255.
            img_bch.append(crop_img)
        out_put = sess.run(out, feed_dict={tfx: img_bch})
        for num in range(BATCH):
            out_labels = out_put[num].argmax(axis=1)
            out_str = ''.join(str(s) for s in out_labels if s != 10)
            df.loc[idx*BATCH + num] = [all_test_names[idx*BATCH + num], out_str]
        print('idx--all', idx, '--', test_time)
        # if idx > 30:
        #     break
    df.to_csv("./csv/sample_submit_0528.csv", index=None)


    #     img = load_test_img_by_name(test_pic)
    #     out_put = sess.run(out, feed_dict={tfx: img})
    #     out_labels = out_put[0].argmax(axis=1)
    #     out_str = ''.join(str(s) for s in out_labels if s != 10)
    #     df.loc[idx] = [test_pic, out_str]
    #     # cv2.imwrite('./data/vgg_res/'+out_str+'_'+test_pic, img)
    #     if idx % 10 == 0:
    #         print(idx, "finish")
    #     if idx > 300:
    #         break
    # df.to_csv("./csv/sample_submit_0525.csv", index=None)





def load_test_img_by_name(img_name):

    TEST_DATA = './data/my_data/test/'

    img0 = cv2.imread(TEST_DATA + img_name)
    if img0 is None:
        print('PIC NOT FOUND')
    # resize_img, _, _, _ = letterbox_resize(img0, 224, 224)
    resize_img = resize(img0, (64, 128))
    a, b = np.random.randint(0, 4), np.random.randint(0, 8)
    crop_img = resize_img[a:a + 60, b:b + 120, :]
    crop_img = np.asarray(crop_img, np.float32)
    crop_img = crop_img / 255.
    ret_img = crop_img[None, :, :, :]
    return ret_img


def parse_labels(old_labels):
    ret_labels = []
    for label in old_labels:
        label = label[:5]
        padding = np.zeros(shape=[5, 11])
        for idx, num in enumerate(label):
            padding[idx][num] = 1
        padding[len(label):, -1] = 1
        ret_labels.append(padding)
    return ret_labels


def get_image_and_label(phase):
    image_label_dict = {}
    if phase == 'train':
        for line in open('../data/train.txt').readlines():
            line_idx, pic_path, boxes, labels, img_width, img_height = parse_line(line)
            # print(line_idx, pic_path, boxes, labels, img_width, img_height)
            # image_name = pic_path.strip().split('/')[-1]
            image_label_dict[line_idx] = [pic_path, labels]
    else:
        for line in open('../data/val.txt').readlines():
            line_idx, pic_path, boxes, labels, img_width, img_height = parse_line(line)
            # print(line_idx, pic_path, boxes, labels, img_width, img_height)
            # image_name = pic_path.strip().split('/')[-1]
            image_label_dict[line_idx] = [pic_path, labels]
    return image_label_dict


def load_img_labels_by_name(name_idx, phase):
    image_label_dict = get_image_and_label(phase=phase)
    batch_img = []
    batch_labels = []
    # names = []
    for img_id in name_idx:
        img0 = cv2.imread('.'+image_label_dict[img_id][0])
        # names.append(image_label_dict[img_id][0])
        img0 = do_img_aug(img0)
        if img0 is None:
            print('PIC NOT FOUND')
        resize_img = resize(img0, (36, 110))
        a, b = np.random.randint(0, 4), np.random.randint(0, 10)
        crop_img = resize_img[a:a + 32, b:b + 100, :]
        # resize_img, _, _, _ = letterbox_resize(img0, 224, 224)
        # img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        crop_img = np.asarray(crop_img, np.float32)
        crop_img = crop_img / 255.
        batch_img.append(crop_img)
        batch_labels.append(image_label_dict[img_id][1])
    # print('names', names)
    return np.array(batch_img), batch_labels


# train()
# test()
# test_batch()




