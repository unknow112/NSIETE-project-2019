import tensorflow.keras  as keras
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, BatchNormalization, Flatten
from tensorflow.keras.activations import tanh
import numpy as np
from functools import reduce

class Discriminator(keras.models.Sequential):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.first_compile = True
        self.trainable = False

        conv = lambda K,N,S : Conv2D(
            padding='same',
            filters=N,
            kernel_size=K,
            strides=S
        )

        residual_block = lambda *convargs: [
            conv(*convargs),
            BatchNormalization(),
            LeakyReLU(alpha=0.2)
        ]

        self.model = [
            conv(3,64,1),
            LeakyReLU(alpha=0.2),
            *residual_block(3,64,2),
            *residual_block(3,128,1),
            *residual_block(3,128,2),
            *residual_block(3,256,1),
            *residual_block(3,256,2),
            *residual_block(3,512,1),
            *residual_block(3,512,2),
            Dense(1024),
            LeakyReLU(alpha=0.2),
            Flatten(), # wild guess, try maybe before first dense
            Dense(1,activation=tanh)
        ]
        for layer in self.model:
            self.add(layer)

        
    def train_on_batch(self, *args):
        self.trainable = True
        self.compile()
        super().train_on_batch(*args)
        self.trainable = False
        self.compile()


    def compile(self, *args, **kwargs):
        if self.first_compile:
            self.compile_args = args
            self.compile_kwargs = kwargs
            self.first_compile = False

        super().compile(*self.compile_args, **self.compile_kwargs)

        
"""

    def call(self, x):
        return reduce(lambda partial,layer: layer(partial), self.model, x)
"""
    
