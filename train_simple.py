import os
import numpy as np
import tensorflow as tf
import random
import config as cf
import warnings
warnings.filterwarnings('ignore')


def dataloder(img_root, info):
    img_paths = []
    labels = []
    for ele in info:
        line = ele.split(';')
        img_name = line[0]
        img_class = int(line[1][:-1])
        img_path = os.path.join(img_root, img_name)
        img_paths.append(img_path)
        labels.append(img_class)
    return img_paths, labels


def train_dataprocess(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片

    image_emhance = tf.image.random_brightness(image_decoded, 0.3)
    image_emhance = tf.image.random_contrast(image_emhance, 0.7, 1.3)
    image_resized = tf.image.resize(image_emhance, cf.resize_shape) / 255.0

    label = tf.cast(label, dtype=tf.int64)
    return image_resized, label


def val_dataprocess(filename, label):
    image_string = tf.io.read_file(filename)
    image_emhance = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_emhance, cf.resize_shape) / 255.0
    label = tf.cast(label, dtype=tf.int64)
    return image_resized, label


def make_dataset(img_root, txt_path, batch_size):
    info = []
    with open(txt_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            info.append(line)
        random.shuffle(info)

    train_info = info[:int(len(info) * cf.train_and_val)]
    val_info = info[int(len(info) * cf.train_and_val):]

    steps_per_epoch = len(train_info) // batch_size
    validation_steps = len(val_info) // batch_size

    train_filenames, train_labels = dataloder(img_root, train_info)
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=train_dataprocess,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size).repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_filenames, val_labels = dataloder(img_root, val_info)
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(
        map_func=val_dataprocess,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, steps_per_epoch, validation_steps


if __name__ == '__main__':

    train_dataset, val_dataset, steps_per_epoch, validation_steps = make_dataset(cf.train_img, cf.train_txt, cf.batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=cf.h5_savePath, save_freq=cf.save_freq),
        tf.keras.callbacks.TensorBoard(log_dir=cf.tensorboard_path, update_freq='batch')]

    if cf.base_model == 'Vgg16':
        model = tf.keras.applications.VGG16(weights=cf.premodel_path, include_top=False, pooling='avg')
        model.trainable = cf.trainable
        inputs = tf.keras.layers.Input(shape=cf.input_shape)
        x = model(inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(units=cf.num_classes, activation='softmax')(x)

    elif cf.base_model == 'ResNet50':
        model = tf.keras.applications.ResNet50(weights=cf.premodel_path, include_top=False, pooling='avg')
        model.trainable = cf.trainable
        inputs = tf.keras.layers.Input(shape=cf.input_shape)
        x = model(inputs)
        x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(units=cf.num_classes, activation='softmax')(x)

    else:
        inputs = tf.keras.layers.Input(shape=cf.input_shape)
        conv_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        maxpool_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_1)
        conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(maxpool_1)
        maxpool_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_2)
        conv_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)
        maxpool_3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_3)
        conv_4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(maxpool_3)
        maxpool_4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_4)
        flatten = tf.keras.layers.Flatten()(maxpool_4)
        dropout_1 = tf.keras.layers.Dropout(0.5)(flatten)
        dense_1 = tf.keras.layers.Dense(1024, activation='relu')(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(0.5)(dense_1)
        dense_2 = tf.keras.layers.Dense(256, activation='relu')(dropout_2)
        outputs = tf.keras.layers.Dense(units=cf.num_classes, activation='softmax')(dense_2)

    my_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    my_model.compile(optimizer=tf.keras.optimizers.Adam(lr=cf.base_learning_rate),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

    my_model.summary()

    my_model.fit(train_dataset,
                 steps_per_epoch=steps_per_epoch,
                 epochs=cf.epochs,
                 validation_data=val_dataset,
                 validation_steps=validation_steps,
                 callbacks=callbacks)

    tf.saved_model.save(my_model, cf.pb_savepath)
