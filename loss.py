from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as kb

vgg = VGG19()

def content_loss(yp, yt): #unfininshed
    vyp = vgg.predict(yp)
    vyt = vgg.predict(yt)
    print("OK")
    return kb.sqrt(kb.sum(kb.square(vyp - vyt), axis=1))


def adversarial_loss(yp,_):
    return kb.sum( kb.log(yp) * -1) 