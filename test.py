import os
import numpy as np
import tensorflow as tf
import random
import yaml
import time
import warnings
warnings.filterwarnings('ignore')


def dataloder(img_root, txt_path):
    img_paths = []
    img_labels = []
    with open(txt_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            line = line.split(';')
            print(line)
            img_name = line[0]
            img_class = int(line[1][:-1])
            img_path = os.path.join(img_root, img_name)
            img_paths.append(img_path)
            img_labels.append(img_class)
    return img_paths, img_labels


def decode_and_resize(file_path, img_label):
    image_string = tf.io.read_file(file_path)
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, args['train']['data_loader']['resize_shape']) / 255.0

    img_label = tf.cast(img_label, dtype=tf.int64)
    return image_resized, img_label, file_path


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config_path = './config.yml'

    with open(config_path, 'r') as f:
        args = yaml.load(f.read())

    model = tf.saved_model.load(args['test']['pb_load_path'])

    test_filenames, test_labels = dataloder(args['test']['test_img'], args['test']['test_txt'])

    test_filenames = tf.constant(test_filenames)
    test_labels = tf.constant(test_labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(
        map_func=decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(args['train']['data_loader']['batch_size'])
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    cnt = 0
    correct = 0
    sum_time = 0
    wrong_imgPaths = []
    for step, (images, labels, filepath) in enumerate(test_dataset):
        start = time.time()
        preds = model(images)
        end = time.time()
        sum_time += (end - start)

        preds = preds.numpy()
        labels = labels.numpy()
        paths = filepath.numpy()
        for i, pred in enumerate(preds):
            cnt += 1
            if np.argmax(pred) == labels[i]:
                correct += 1
            else:
                wrong_imgPaths.append(paths[i])
    print('correct:{}/{}'.format(correct, cnt))
    print('acc: ', correct / cnt)
    print('pred_time: ', sum_time / cnt)
    print('wrong images paths: ')
    for path in wrong_imgPaths:
        print(path)
