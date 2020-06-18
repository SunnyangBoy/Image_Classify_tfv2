import os
import numpy as np
import tensorflow as tf
import config as cf
import yaml


def dataloder(img_root):
    img_paths = []
    for path in sorted(os.listdir(img_root)):
        img_path = os.path.join(img_root, path)
        img_paths.append(img_path)
    return img_paths


def decode_and_resize(file_path):
    image_string = tf.io.read_file(file_path)
    image_decoded = tf.image.decode_png(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, args['train']['data_loader']['resize_shape']) / 255.0
    return image_resized, file_path


if __name__ == '__main__':
    config_path = './config.yml'

    with open(config_path, 'r') as f:
        args = yaml.load(f.read())

    model = tf.saved_model.load(args['test']['pb_load_path'])

    test_filenames = dataloder(args['infer']['infer_img'])

    test_filenames = tf.constant(test_filenames)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames))
    test_dataset = test_dataset.map(
        map_func=decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(1)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for step, (images, filepaths) in enumerate(test_dataset):
        preds = model(images)
        preds = preds.numpy()
        paths = filepaths.numpy()
        for i, pred in enumerate(preds):
            print('pred class: ', np.argmax(pred))
            print('file path: ', paths[i])
            print('\n')

