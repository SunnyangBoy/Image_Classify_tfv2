from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import math
import os.path as osp
import yaml
import tensorflow.keras.backend as K
from tensorflow.python.framework import graph_util, graph_io


if __name__ == '__main__':
    config_path = './config.yml'

    with open(config_path, 'r') as f:
        args = yaml.load(f.read())

    # 路径参数
    h5_dir = args['train']['save_model']['pb_save_path']
    h5_file = 'model_batch120.h5'
    h5_file_path = os.path.join(h5_dir, h5_file)

    # 加载模型
    h5_model = load_model(h5_file_path, compile=False)
    h5_model.summary()

    tf.saved_model.save(h5_model, os.path.join(args['train']['save_model']['pb_save_path'], 'frozen_model'))
