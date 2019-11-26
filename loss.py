from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as kb

vgg = VGG19()

def content_loss(yp, yt): #unfininshed
    vyp = vgg.predict(yp)
    vyt = vgg.predict(yt)
    return kb.sqrt(kb.sum(kb.square(vyp - vyt), axis=1))


def adversarial_loss(yp, _): #ditto
    _,propability = yp
    return kb.sum( kb.log(propability) * -1) 