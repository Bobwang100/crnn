#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text of a single image
"""
import argparse
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glog as logger
import wordninja
import os
from config import global_config
from crnn_model import crnn_net
# from data_provider import tf_io_pipline_fast_tools
from skimage.transform import resize
CFG = global_config.cfg
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
all_test_imgs = [pic for pic in os.listdir('../data/my_data/test/')]
TEST_BATCH = 64
import pandas as pd

def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='../data/my_data/train/012430.png')
    parser.add_argument('--weights_path', type=str, default='model/crnn_syn90k/shadownet_2020-06-16-16-59-43.ckpt-235000',
                        help='Path to the pre-trained weights to use')
    parser.add_argument('-c', '--char_dict_path', type=str, default='../data/char_dict/char_dict_en.json',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str, default='../data/char_dict/ord_map_en.json',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=True,
                        help='Whether to display images')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path, is_vis, is_english=False):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_vis:
    :param is_english:
    :return:
    """
    df = pd.DataFrame({"file_name": [], "file_code": []})
    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[TEST_BATCH, CFG.ARCH.INPUT_SIZE[1], CFG.ARCH.INPUT_SIZE[0], CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )
    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )

    decodes, _ = tf.nn.ctc_greedy_decoder(
        inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(TEST_BATCH),
        merge_repeated=True
    )
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)
    saver = tf.train.Saver()
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
    for batch_num in range(int(40000/TEST_BATCH)):
        batch_images_names = []
        all_images = []
        for pic in all_test_imgs[batch_num*TEST_BATCH:(batch_num+1)*TEST_BATCH]:
            image = cv2.imread('../data/my_data/test/'+pic, cv2.IMREAD_COLOR)
            batch_images_names.append(pic)
            # print('picture name is ', pic)
            # image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            image = resize(image, (64, 100))
            image_vis = image
            image = np.asarray(image, np.float32)
            # image = crop_img / 255.
            all_images.append(image)

        # if batch_num > 2:
        #     break
        # codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        #     char_dict_path=char_dict_path,
        #     ord_map_dict_path=ord_map_dict_path
        # )
        preds = sess.run(decodes, feed_dict={inputdata: all_images})
        # print('before codes:', preds)
        result_list = []
        for i in range(TEST_BATCH):
            result = preds[0].values[preds[0].indices[:, 0] == i]
            result_str = ''.join(str(i) for i in result)
            df.loc[TEST_BATCH*batch_num + i] = [batch_images_names[i], result_str]
        # if batch_num > 2:
        #     break
        print('finish batch %d total batch %d' %(batch_num, 40000/TEST_BATCH))
            # result_list.append(result_str)
    df = df.sort_values('file_name')
    df.to_csv('./submit0617.csv', index=None)


            # preds = codec.sparse_tensor_to_str(preds[0])[0]
            # print('after codec:', preds)
            # if is_english:
            #     preds = ' '.join(wordninja.split(preds))
            #
            # logger.info('Predict image {:s} result: {:s}'.format(
            #     ops.split(image_path)[1], preds)
            # )
            #
            # if is_vis:
            #     plt.figure('CRNN Model Demo')
            #     plt.imshow(image_vis[:, :, (2, 1, 0)])
            #     plt.show()

    sess.close()

    return


if __name__ == '__main__':
    """
    
    """
    # init images
    args = init_args()

    # detect images
    recognize(
        image_path=args.image_path,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        is_vis=args.visualize
    )
