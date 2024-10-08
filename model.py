'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2024/10/08
'''
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

class resnet20 ():
    def __init__ (self, weight_decay, num_classes, ratio):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.initializer = tf.keras.initializers.GlorotUniform(seed = int(time.time()))
        self.num_classes = num_classes
        self.ratio = ratio

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters,
                   (3, 3),
                   strides = strides,
                   padding = "same",
                   use_bias = False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = Conv2D(num_filters,
                   (3, 3),
                   padding = "same",
                   use_bias = False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(gamma_initializer = 'zeros')(x)
        if projection:
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              padding = "same",
                              use_bias = False,
                              kernel_initializer = self.initializer,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              strides = strides,
                              padding = "same",
                              use_bias = False,
                              kernel_initializer = self.initializer,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        x = Conv2D(16,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer) (x_in)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (3):
            if i == 0:
                x = self.res_block(x, int(16 * self.ratio), projection = True)
            else:
                x = self.res_block(x, int(16 * self.ratio))

        for i in range (3):
            if i == 0:
                x = self.res_block(x, int(32 * self.ratio), strides = (2, 2))
            else:
                x = self.res_block(x, int(32 * self.ratio))

        for i in range (3):
            if i == 0:
                x = self.res_block(x, int(64 * self.ratio), strides = (2, 2))
            else:
                x = self.res_block(x, int(64 * self.ratio))

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        y = Dense(self.num_classes, name='fully_connected',
                  kernel_initializer = self.initializer,
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "resnet20")

