import tensorflow.keras  as keras
from tensorflow.keras.layers import Conv2D, Dense, PReLU, BatchNormalization, Flatten, add, UpSampling2D, Wrapper
from tensorflow.keras.activations import sigmoid

from functools import reduce


class SkipAdder(keras.layers.Layer):
    def __init__(self, model):
        super(SkipAdder, self).__init__()
        self.model = model

    def build(self, arg):
        for element in self.model:
            element.build(arg)
        

    def call(self, x):
        return add([
            x, 
            reduce(lambda partial,layer: layer(partial), self.model, x)
        ])

class Generator(keras.Model):
    def __init__(self, residual_block_count=16):
        super(Generator, self).__init__()

        conv = lambda K,N,S : Conv2D(
            padding='same',
            filters=N,
            kernel_size=K,
            strides=S
        )

        residual_block = lambda *convargs: SkipAdder([
            conv(*convargs),
            BatchNormalization(),
            PReLU(),
            conv(*convargs),
            BatchNormalization()
        ])

        self.model = [
            conv(9,64,1),
            PReLU(),
            SkipAdder([
                *[residual_block(3,64,1)] * residual_block_count,
                conv(3,64,1),
                BatchNormalization(),                    
            ]),
            *[conv(3,256,1),
            UpSampling2D(size=2),
            PReLU()] * 2,
            conv(9,3,1)
        ]

    def call(self, x):
        return reduce(lambda partial,layer: layer(partial), self.model, x)