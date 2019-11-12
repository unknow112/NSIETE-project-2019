import tensorflow.keras  as keras
from tensorflow.keras.layers import Conv2D, Dense, PReLU, BatchNormalization, Flatten, add, UpSampling2D, Activation
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

        upsampling_block = lambda :[
            conv(3,256,1),
            UpSampling2D(),
            PReLU()
        ]

        self.model = [
            conv(9,64,1),
            PReLU(),
            SkipAdder([
                *[
                    residual_block(3,64,1)
                    for _ in range(residual_block_count)
                 ],
                conv(3,64,1),
                BatchNormalization(),                    
            ]),
            *upsampling_block(),
            *upsampling_block(),
            conv(9,3,1),
            Activation('tanh')
        ]

    def call(self, x):
        return reduce(lambda partial,layer: layer(partial), self.model, np.array([x]))
