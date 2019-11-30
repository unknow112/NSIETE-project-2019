"""
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as kb
from tensorflow.python.ops import math_ops

vgg = VGG19()

def content_loss(yp, yt): #unfininshed
    return kb.sqrt(math_ops.reduce_sum(kb.square(vgg(yp) - vgg(yt))))


def adversarial_loss(yp,_):
    return kb.sum( kb.log(yp) * -1)
"""

from tensorflow.keras.losses import MSE
from tensorflow.keras.backend import mean


class GeneratorLoss:
    def __init__(self, discriminator):
        self.d = discriminator

    def __call__(self, y_true, y_pred, *_, **__):
        return mean([
            MSE(y_true, y_pred),
            MSE(self.d(y_true), self.d(y_pred))
        ])
