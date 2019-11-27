from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as kb
from tensorflow.python.ops import math_ops

vgg = VGG19()

def content_loss(yp, yt): #unfininshed
    return kb.sqrt(math_ops.reduce_sum(kb.square(vgg(yp) - vgg(yt))))


def adversarial_loss(yp,_):
    return kb.sum( kb.log(yp) * -1) 