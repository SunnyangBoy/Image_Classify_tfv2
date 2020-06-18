import os
import numpy as np
import tensorflow as tf
import random
import yaml
import warnings
warnings.filterwarnings('ignore')


class Trainer:

    def __init__(self, config_path, config_load_mode='yaml'):
        self.args = {}
        if config_load_mode == 'yaml':
            self.args = self.load_config_from_yaml(config_path)
        elif config_load_mode == 'db':
            self.args = self.load_config_from_db()

        print(self.args)

        self.model = self.build_model()
        self.train_data, self.val_data, self.train_steps_epoch, self.val_steps = self.build_dataset()

    @staticmethod
    def load_config_from_yaml(config_path):
        with open(config_path, 'r') as f:
            configs = yaml.load(f.read())
            return configs

    @staticmethod
    def load_config_from_db():
        return {}

    def build_model(self, ):
        args = self.args['train']['base_model']
        if args['model_name'] == 'Vgg16':
            model = tf.keras.applications.VGG16(weights=args['pretrain_model'], include_top=False, pooling='avg')
            model.trainable = args['trainable']
            inputs = tf.keras.layers.Input(shape=args['input_shape'])
            x = model(inputs)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(units=args['num_classes'], activation='softmax')(x)

        elif args['model_name'] == 'ResNet50':
            model = tf.keras.applications.ResNet50(weights=args['pretrain_model'], include_top=False, pooling='avg')
            model.trainable = args['trainable']
            inputs = tf.keras.layers.Input(shape=args['input_shape'])
            x = model(inputs)
            x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(units=args['num_classes'], activation='softmax')(x)

        else:
            inputs = tf.keras.layers.Input(shape=args['input_shape'])
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
            dense_2 = tf.keras.layers.Dense(256, activation='relu')(dense_1)
            outputs = tf.keras.layers.Dense(units=args['num_classes'], activation='softmax')(dense_2)

        my_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return my_model

    def build_dataset(self, ):
        args = self.args['train']['data_loader']
        info = []
        with open(args['train_txt'], 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                info.append(line)
            random.shuffle(info)

        train_info = info[:int(len(info) * args['train_and_val'])]
        val_info = info[int(len(info) * args['train_and_val']):]

        train_steps_epoch = len(train_info) // args['batch_size']
        val_steps = len(val_info) // args['batch_size']

        train_filenames, train_labels = self.data_loader(args['train_img'], train_info)
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(
            map_func=self.train_data_process,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(args['batch_size']).repeat()
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_filenames, val_labels = self.data_loader(args['train_img'], val_info)
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(
            map_func=self.val_data_process,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(args['batch_size']).repeat()
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset, train_steps_epoch, val_steps

    def train(self, ):
        save_args = self.args['train']['save_model']
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=save_args['h5_savePath'], save_freq=save_args['save_freq']),
            tf.keras.callbacks.TensorBoard(log_dir=self.args['train']['log_dir'], update_freq='batch')]

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.args['train']['learning_rate']),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

        self.model.fit(self.train_data,
                       steps_per_epoch=self.train_steps_epoch,
                       epochs=self.args['train']['epochs'],
                       validation_data=self.val_data,
                       validation_steps=self.val_steps,
                       callbacks=callbacks)

        tf.saved_model.save(self.model, save_args['pb_save_path'])

    @staticmethod
    def data_loader(img_root, info):
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

    def train_data_process(self, filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片

        image_emhance = tf.image.random_brightness(image_decoded, 0.3)
        image_emhance = tf.image.random_contrast(image_emhance, 0.7, 1.3)
        image_resized = tf.image.resize(image_emhance, self.args['train']['data_loader']['resize_shape']) / 255.0

        label = tf.cast(label, dtype=tf.int64)
        return image_resized, label

    def val_data_process(self, filename, label):
        image_string = tf.io.read_file(filename)
        image_emhance = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
        image_resized = tf.image.resize(image_emhance, self.args['train']['data_loader']['resize_shape']) / 255.0
        label = tf.cast(label, dtype=tf.int64)
        return image_resized, label


if __name__ == '__main__':
    trainer = Trainer(config_path='./config.yml')
    trainer.train()
