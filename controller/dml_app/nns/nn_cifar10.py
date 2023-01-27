'''
替换了学长原来的网络结构
牺牲了5%的精度，以减小模型大小，加快训练速度
验证集精度为80%（10个epoch）
'''
from tensorflow.keras import Input, layers, Model, losses


class Cifar10 (object):
    def __init__ (self):
        self.input_shape = [-1, 32, 32, 3]  # -1 means no matter how much data
        
        x = Input(shape=(32, 32, 3))
        y = x
        y = layers.Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = layers.Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

        y = layers.Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = layers.Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

        y = layers.Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = layers.Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

        y = layers.Flatten()(y)
        y = layers.Dense(units=128, activation='relu', kernel_initializer='he_normal')(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Dense(units=10, activation='softmax', kernel_initializer='he_normal')(y)
        self.model = Model (x, y)
        self.model.compile (optimizer='adam', loss=losses.SparseCategoricalCrossentropy (from_logits=True),
            metrics=['accuracy'])
        self.size = 4 * self.model.count_params ()  # 4 byte per np.float32


nn = Cifar10 ()

if __name__ == '__main__':
	nn.model.summary ()
